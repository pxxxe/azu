import typer
import asyncio
import aiohttp
import json
import os
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction

app = typer.Typer()

# LOAD CONFIG
API_URL = "http://localhost:8000"
with open(".env") as f:
    env = dict(line.strip().split('=') for line in f if '=' in line)

PLATFORM_WALLET = Pubkey.from_string(env['PLATFORM_WALLET_PUBKEY'])
RPC_URL = env['SOLANA_RPC_URL']

def get_wallet():
    with open(os.path.expanduser("~/.config/solana/id.json")) as f:
        return Keypair.from_bytes(bytes(json.load(f)))

@app.command()
def deposit(amount_sol: float):
    async def _run():
        kp = get_wallet()
        client = AsyncClient(RPC_URL)

        print(f"💳 Sending {amount_sol} SOL...")
        lamports = int(amount_sol * 1_000_000_000)

        # Create transfer instruction using the function, not a class
        ix = transfer(TransferParams(
            from_pubkey=kp.pubkey(),
            to_pubkey=PLATFORM_WALLET,
            lamports=lamports
        ))

        # Get latest blockhash
        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash

        # Create and sign transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            kp.pubkey(),
            [kp],
            blockhash
        )

        # Send transaction
        sig = await client.send_transaction(tx)
        print(f"Tx Sent: {sig.value}")

        # Notify Backend
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{API_URL}/deposit", json={
                "tx_sig": str(sig.value),
                "user_pubkey": str(kp.pubkey())
            }) as resp:
                print(await resp.json())

    asyncio.run(_run())

@app.command()
def prompt(text: str, model: str = "Qwen/Qwen2.5-0.5B"):
    async def _run():
        kp = get_wallet()
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{API_URL}/submit", json={
                "user_pubkey": str(kp.pubkey()),
                "model_id": model,
                "prompt": text
            }) as resp:
                print(await resp.json())

    asyncio.run(_run())

if __name__ == "__main__":
    app()
