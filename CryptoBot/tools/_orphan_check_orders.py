import requests
h = {'APCA-API-KEY-ID': 'PKFYHFB2A7EJEXQUKOEHCYLNR2', 'APCA-API-SECRET-KEY': '2397JkVebdYJrVaVUobiJAqFWzYXA3o94tBXEJC83Y5S'}
orders = requests.get('https://paper-api.alpaca.markets/v2/orders?status=open', headers=h).json()
print(f'Pending orders: {len(orders)}')
for o in sorted(orders, key=lambda x: x['symbol']):
    print(f"  {o['symbol']:30s} {o['side']:>4s} qty={o['qty']:>3s} type={o['type']} limit={o.get('limit_price','N/A')} status={o['status']}")
