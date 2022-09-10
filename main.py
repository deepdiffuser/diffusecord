#!/usr/bin/env python

import discord
import json
import os
import time
import diffusers
import io
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from discord.ext import commands


# CONFIGURATION
device = "cuda"

if os.environ["CUDA_DEVICE"] != "":
    device = "cuda:" + os.environ["CUDA_DEVICE"]


class DummySafetyChecker():
    def __init__(self, *args, **kwargs):
        # required monkeypatching to prevent an error splitting the module name to check type
        self.__module__ = "foo.bar.foo.bar"

    def __call__(self, images, **kwargs):
        return (images, False)


async def txt2img(sd_pipeline, prompt: str, seed: int, scale: float, steps: int, height: int, width: int):
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, sd_pipeline.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )

    with autocast("cuda"):
        image = sd_pipeline(prompt=prompt, width=width, height=height,
                            guidance_scale=scale, num_inference_steps=steps, latents=latents).images[0]

        imbytes = io.BytesIO()
        image.save(imbytes, format='PNG')
        imbytes.seek(0)
        return imbytes


pipe = None
if os.environ["DISABLE_CENSORSHIP"] == "1":
    # fp16 is half precision
    pipe = StableDiffusionPipeline.from_pretrained(
        "./stable-diffusion-v1-4", local_files_only=True, use_auth_token=False,  revision="fp16",
        torch_dtype=torch.float16, safety_checker=DummySafetyChecker())
else:
    # fp16 is half precision
    pipe = StableDiffusionPipeline.from_pretrained(
        "./stable-diffusion-v1-4", local_files_only=True, use_auth_token=False,  revision="fp16",
        torch_dtype=torch.float16)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

bot = discord.Bot()


@bot.slash_command()
@commands.cooldown(1, 5, commands.BucketType.user)
@commands.max_concurrency(1)
async def imagine(ctx, prompt: str,
                  seed: int = None, scale: float = 7.0, steps: int = 50, height: int = 512, width: int = 512):

    if seed is None:
        seed = int(time.time())

    if width > 1024 or height > 1024:
        await ctx.respond("error: too big, max 1024")
        return

    if width < 64 or height < 64:
        await ctx.respond("error: too small, min 64")
        return

    if scale < 0.0:
        await ctx.respond("error: scale must be positive")
        return

    if steps > 100:
        await ctx.respond("error: too many steps, max 100")
        return

    # change width and height to nearest multiple of 64
    width = 64 * (width // 64)
    height = 64 * (height // 64)

    await ctx.defer()

    try:
        img = await txt2img(pipe,
                            prompt,
                            seed,
                            scale,
                            steps,
                            height,
                            width)
        res = json.dumps({"prompt": prompt, "width": width, "height": height,
                          "scale": scale, "steps": steps, "seed": seed})

        pic = discord.File(img, "image.png")
        await ctx.respond(res, file=pic)
    except:
        await ctx.respond("error during generation, probably out of memory")


token = os.environ["DISCORD_TOKEN"]
bot.run(token)
