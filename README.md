# diffusecord

A super-simple stable diffusion Discord bot.

## Installation

1. Install Docker
2. Install Docker NVIDIA drivers (nvidia-docker2 on linux)
3. Clone the `diffusers` version of stable diffusion to ./stable-diffusion-v1-4 (or download at https://archive.org/download/stable-diffusion-v1-4.tar)
4. Put your discord bot token in a new file `.env_docker`, with `DISCORD_TOKEN=xxx`
5. `docker compose up --build`

