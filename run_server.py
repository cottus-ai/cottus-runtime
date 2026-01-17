import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Cottus API Server")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    from cottus.model import load_hf_model
    from cottus.async_engine import AsyncEngine
    from cottus.server import CottusServer

    print(f"Loading model: {args.model}")
    engine_config, weight_ptrs, tokenizer = load_hf_model(args.model, device=args.device)

    async_engine = AsyncEngine(engine_config, weight_ptrs)
    async_engine.start()

    server = CottusServer(async_engine, tokenizer)

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
