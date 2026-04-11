import os, sys
sys.path.insert(0, 'cryptotrades')
from dotenv import load_dotenv
load_dotenv('cryptotrades/.env', override=False)
import requests

url = 'https://data.alpaca.markets/v1beta3/crypto/us/latest/trades'
headers = {
    'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY', ''),
    'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET', ''),
}
symbols = [
    'BTC/USD','ETH/USD','SOL/USD','ADA/USD','AVAX/USD','DOGE/USD',
    'LINK/USD','XRP/USD','LTC/USD','UNI/USD','XLM/USD','BCH/USD',
    'DOT/USD','MATIC/USD','ATOM/USD','NEAR/USD','AAVE/USD','PAXG/USD'
]
ok = []
fail = []
for s in symbols:
    try:
        r = requests.get(f'{url}?symbols={s}', headers=headers, timeout=5)
        data = r.json().get('trades', {})
        if s in data and 'p' in data[s]:
            ok.append(s)
            print(f'  OK  {s} = ${data[s]["p"]}')
        else:
            fail.append(s)
            print(f'FAIL  {s} (status {r.status_code})')
    except Exception as e:
        fail.append(s)
        print(f'ERR   {s} - {e}')

print(f'\n{len(ok)} OK, {len(fail)} FAILED')
print(f'Failed: {fail}')
