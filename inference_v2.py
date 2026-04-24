import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import yaml
import soundfile as sf
import time
from modules.commons import str2bool
from modules.v2.trus import TruSManager

# Set up device and torch configurations
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Global variables to store model instances
vc_wrapper_v2 = None


def load_v2_models(args):
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint_path,
        cfm_checkpoint_path=args.cfm_checkpoint_path
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        if hasattr(torch._inductor.config, "fx_graph_cache"):
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()

    return vc_wrapper


def convert_voice_v2(source_audio_path, target_audio_path, args):
    """Convert voice using V2 model"""
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)
    trus_manager = None
    if args.trus_enable:
        trus_manager = TruSManager()
        if args.trus_mode == "record":
            trus_manager.recorder.enabled = True
        elif args.trus_mode == "steer":
            trus_manager.steerer.enabled = True
            if args.trus_profile is None:
                raise ValueError("trus_profile must be provided when trus_mode='steer'")
            profile = torch.load(args.trus_profile, map_location="cpu")
            trus_manager.steerer.load_profile(
                profile,
                alpha_branch={
                    0: args.trus_alpha_branch0,
                    1: args.trus_alpha_branch1,
                    2: args.trus_alpha_branch2,
                }
            )

        vc_wrapper_v2.set_trus_manager(trus_manager)
    else:
        vc_wrapper_v2.set_trus_manager(None)
    # Use the generator function but collect all outputs
    generator = vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=True,
        trus_manager=trus_manager,
    )

    # Collect all outputs from the generator
    full_audio = None
    for output in generator:
        _, full_audio = output

    if trus_manager is not None and args.trus_mode == "record" and args.trus_cache_out is not None:
        cache_dir = os.path.dirname(args.trus_cache_out)
        if cache_dir != "":
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(trus_manager.recorder.cache, args.trus_cache_out)

    return full_audio
    

def main(args):
    # Create output directory if it doesn't exist
    print("enter main")
   
    os.makedirs(args.output, exist_ok=True)
    
    start_time = time.time()
    converted_audio = convert_voice_v2(args.source, args.target, args)
    end_time = time.time()

    if converted_audio is None:
        print("Error: Failed to convert voice")
        return

    # Save the converted audio
    source_name = os.path.basename(args.source).split(".")[0]
    target_name = os.path.basename(args.target).split(".")[0]

    # Create a descriptive filename
    filename = f"vc_v2_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"

    output_path = os.path.join(args.output, filename)
    save_sr, converted_audio = converted_audio
    sf.write(output_path, converted_audio, save_sr)

    print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source audio file")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target/reference audio file")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for converted audio")
    parser.add_argument("--diffusion-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)")
    parser.add_argument("--compile", type=bool, default=False,
                        help="Whether to compile the model for faster inference")

    # V2 specific arguments
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7,
                        help="Intelligibility CFG rate for V2 model")
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7,
                        help="Similarity CFG rate for V2 model")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter for V2 model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature sampling parameter for V2 model")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for V2 model")
    parser.add_argument("--convert-style", type=str2bool, default=False,
                        help="Convert style/emotion/accent for V2 model")
    parser.add_argument("--anonymization-only", type=str2bool, default=False,
                        help="Anonymization only mode for V2 model")

    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--trus-enable", type=str2bool, default=False,
                        help="Enable TruS activation recording/steering")
    parser.add_argument("--trus-mode", type=str, default="off",
                        choices=["off", "record", "steer"],
                        help="TruS mode")
    parser.add_argument("--trus-profile", type=str, default=None,
                        help="Path to saved TruS steering profile (.pt)")
    parser.add_argument("--trus-alpha-branch0", type=float, default=0.4,
                    help="TruS alpha for cond_txt_spk branch")
    parser.add_argument("--trus-alpha-branch1", type=float, default=0.0,
                    help="TruS alpha for cond_txt branch")
    parser.add_argument("--trus-alpha-branch2", type=float, default=0.0,
                    help="TruS alpha for uncond branch")
    parser.add_argument("--trus-cache-out", type=str, default=None,
                        help="Path to save recorded activation cache (.pt)")
    args = parser.parse_args()
    main(args)