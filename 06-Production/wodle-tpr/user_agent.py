import requests
import random
import time
import argparse
import os
import re

# ==========================
# CONFIGURAÇÃO
# ==========================

API_URL = [
    "https://api.infraspeak.com/v3/elements",
    "https://api.infraspeak.com/v3/locals/all",
    "https://api.infraspeak.com/v3/operators",
    "https://api.infraspeak.com/v3/events"
]

# Lista de BEARER_TOKEN é deixada vazia por segurança; use --tokens-file para carregar tokens a partir de `access_tokens.txt`.
BEARER_TOKEN = [
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiMTEyMWQ4ODJiMTNhMWNjYmI3MGY2ZGEwZTRiOTdlY2YzMDE1MTY5MzVkOTM1NTMwYmJiY2UzZDI0NjM2NTAwNDNiZmE0YmEwMTczMTBlYmYiLCJpYXQiOjE3NjYwMzQyMzQsIm5iZiI6MTc2NjAzNDIzNCwiZXhwIjoxNzcxMjE4MjM0LCJzdWIiOiIyNjgxNCIsInNjb3BlcyI6WyIqIl0sInV0eXBlIjoiT1BFUkFUT1IiLCJ1dHlwZV9pZCI6MjExMzgsIm5hbWUiOiJEYW5pZWwgQWRtaW4gMjE0IiwiZW1haWwiOiJkYW5pZWwubWFjaWVsK2FkbWluMjE0QGluZnJhc3BlYWsuY29tIiwiem9uZWluZm8iOiJFdXJvcGUvTGlzYm9uIn0.aZ8Xx7iFfOuwWb9asmp99PE_0RAvNcVEvUqklyP-9KyU7E-lfwatwqede_zSyiWrQJyC-RGpAfAM_pg5Pkz6hPXJYztlVfcgLcIdCJ87MVU021pRqdToXCbzaLWH-EoX26Uu3ZdaeKnCcDJF2zbguS2SjFYpJYiK5pHEJr14U920sh-oqBA-V5hXnafpvlLx1hhtE84b493E3zsHLLcF_y1cGqW37h65-FPzh-5wMU3VRGOZ3MVnNlifK9ebzsl3Fg60tYZqBgfqTm_rzegVwOohiA_e4tH5_8tKa68dZi93bzs0BtL7Ci7yKMJpbUEbmvsFV7i1_CJdJ1bNW15Bd9iqQovyGorbxhuSMPoDoeNtU1s6EyGkUqKkIlqOPze32SsolRNRYePZlg01HoDENBJyv6-lecOykRwGo_KOImVP8TwRni9TxwYX82RXzkszexSOpkb-WdhP3d-Pn__QyExTFGtsDTWmMjA6iC3SkZo32yTOplr8edB1w3ViwpDMXDovk01dU_VXj_yk94psHe9qm4c1gYdgkMFD3kjAN8d4d4xvWdyz5CPVQnOqH1CykhFCLlSM3jD41KMRrqQsXLo8xo8sxMGxOAT66Yd2WCTb6nxhJkU4no61tjqM5UyON2ElrMrTLHVHn5aJtlJz1mfrHOxXqNIJ-BACEIslYxg",
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzMjYiLCJqdGkiOiJhMmIyNTYyNzYxMWMwZTE2NDk4YWE2NjUzOWFjODhiNzcyYmNlYTZiOWYzYWQwY2YwYTkxOWE1MzNiMWFjMGYxODg2MjA1OWM3MTFjZjMzZCIsImlhdCI6MTc2NjAzNzA0NywibmJmIjoxNzY2MDM3MDQ3LCJleHAiOjE3NzEyMjEwNDcsInN1YiI6IjcyNzExIiwic2NvcGVzIjpbIioiXSwidXR5cGUiOiJPUEVSQVRPUiIsInV0eXBlX2lkIjo2MzQ4MCwibmFtZSI6IlRoYWlzIE5ldG8iLCJlbWFpbCI6InRoYWlzLm5ldG8rdGVjaG5pY2lhbjIxNEBpbmZyYXNwZWFrLmNvbSIsInpvbmVpbmZvIjoiRXVyb3BlL0xvbmRvbiJ9.oJvbERpSAa6h4u_67mEXtXPDQZnZHFB-ijk8J5zJXaxeNCXs_5Gw28wqTPICu-sRfYZScp8Jvfhrr3l1gE59xqsX16u0omS1l0BoJZisJbU4Wu-0-9UZMG-X5RoZfh4ZAo06quiF9F3AS_mCyZ-WDrpuT2Rx1WKSi422g6IArEjrBFVRgoj1CRz5Xr9cYCfPO_ohvguBU32NNIqmLVWAggxxxQd2JezsZtc_vdlP6EIWZLhBAf962ONEDGXcLioUT_FmPH-TO_nSrRB1TRwGOUwCAnyc4IJXF3jXb9QD_qdpiUAaLcE5cAhdJSraig299l2GRQ8JUoWPrOCU7FqT9JlYN2wKl8kf1oCYl00Y1zpXnk4dsKL3NUPv1A-vIM3SDazQJAfeHSMPIPSJrrYf3KhsZK4YJplZrQI3uCUVwk8_A4X0IxpZ2Ci7_i3Jy584t3pt3edqCD-5UdYWdtWebPr_xz2G96OP00tCXAK15Zif4hQfCNCIgnE0Ze9r5jDR_tQ2bC0mmL8DCy9CU-5sZQgGPBXWH-QiWP-_n3TS_WnojTkgkHVvJY_yySzQESf6ujvOqOGujsM-yat7ShUb9vjdZvUd-C1dCJwYEz4ZkF7yyIhE1k4dvlR-qWIJMNQ_cabdcbfBN2mzPX4ZydpZYWh7g-UQH1yzpExhMvMLE"
]

USER_AGENTS = [
    # Scanning Tools (mais comuns)
    "sqlmap/1.6.4#stable",
    "sqlmap/1.7.2#stable", 
    "Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)",
    "Nikto/2.5.0",
    "Nikto",
    
    # Proxy & Pentesting Frameworks
    "BurpSuite/2023.1",
    "BurpSuite Professional/2024.1",
    "OWASP ZAP/2.14.0",
    "w3af.org",
    "Metasploit/6.3.0",
    "Metasploit",
    
    # Brute-force Tools
    "Hydra v9.4",
    "Hydra",
    "DirBuster-1.0-RC1",
    "Gobuster/3.6.0",
    "wfuzz/3.1.0",
    
    # Vulnerability Scanners
    "Havij",
    "Acunetix Web Vulnerability Scanner",
    "Nessus/10.5.0",
    "OpenVAS",
    "Nuclei - Open-source project (github.com/projectdiscovery/nuclei)",
    
    # WordPress & CMS Scanners
    "WPScan v3.8.24",
    "Joomla! Scanner",
    
    # Misc Security Tools
    "masscan/1.3.0",
    "curl/7.88.1",  # Muitas vezes usado em scripts maliciosos
    "python-requests/2.31.0",  # Também comum em automação maliciosa
    "Go-http-client/1.1"  # Usado por muitas ferramentas Go
]

# ==========================
# FUNÇÕES UTILITÁRIAS
# ==========================

def load_tokens_from_file(path):
    """Carrega tokens do ficheiro. Procura padrões parecidos com JWT e também aceita tokens simples por linha."""
    tokens = []
    if not os.path.isfile(path):
        return tokens
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip(',').strip('"').strip("'")
            if not line:
                continue
            jwt_match = re.search(r'([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)', line)
            if jwt_match:
                tokens.append(jwt_match.group(1))
            else:
                tokens.append(line)
    return tokens


def generate_fake_jwt():
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    parts = [''.join(random.choices(alphabet, k=20)) for _ in range(3)]
    return '.'.join(parts)


def mask_token(t):
    if not t:
        return ''
    if len(t) <= 12:
        return t[:4] + '...' + t[-4:]
    return t[:6] + '...' + t[-6:]


def random_user_agent(user_agents, mix=False):
    ua = random.choice(user_agents)
    if not mix:
        return ua
    suffix = f" (+r{random.randint(100,999)})"
    ua_modified = re.sub(r'(Chrome/)(\d+)(\.\d+)*', lambda m: f"{m.group(1)}{random.randint(60,140)}", ua)
    return ua_modified + suffix


# ==========================
# FUNÇÃO DE REQUEST
# ==========================

def make_request(urls, tokens, user_agents, mix_ua=False, timeout=15, show_headers='mask', show_response_headers='none'):
    url = random.choice(urls)
    token = random.choice(tokens) if isinstance(tokens, (list, tuple)) and tokens else generate_fake_jwt()
    ua = random_user_agent(user_agents, mix=mix_ua)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": ua,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    print(f"URL: {url}")
    print(f"User-Agent usado: {ua}")

    # Mostrar headers conforme opção (mask por defeito)
    if show_headers == 'full':
        print("Headers:", headers)
    elif show_headers == 'mask':
        h = headers.copy()
        if 'Authorization' in h:
            h['Authorization'] = f"Bearer {mask_token(token)}"
        print("Headers:", h)

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        print(f"Erro na request: {e}")
        return None

    # Mostrar headers da resposta conforme opção
    if response is not None and show_response_headers != 'none':
        try:
            rh = dict(response.headers)
        except Exception:
            rh = {}
        if show_response_headers == 'mask':
            masked = {}
            for k, v in rh.items():
                if k.lower() in ('set-cookie', 'authorization', 'cookie', 'x-auth-token'):
                    masked[k] = mask_token(v if isinstance(v, str) else str(v))
                else:
                    masked[k] = v
            print('Response Headers:', masked)
        else:
            print('Response Headers:', rh)

    print(f"Status Code: {response.status_code}")
    if response.ok:
        try:
            return response.json()
        except Exception:
            return response.text
    else:
        print(response.text)
        return None


# ==========================
# EXECUÇÃO
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enviar requests com URL, Bearer e User-Agent aleatórios')
    parser.add_argument('--count', type=int, default=100, help='Número de requisições a executar')
    parser.add_argument('--min-delay', type=float, default=0.3, help='Delay mínimo entre requests (segundos)')
    parser.add_argument('--max-delay', type=float, default=1.0, help='Delay máximo entre requests (segundos)')
    parser.add_argument('--tokens-file', type=str, default='access_tokens.txt', help='Ficheiro com tokens (um por linha ou texto contendo JWTs)')
    parser.add_argument('--mix-ua', action='store_true', help='Ativa variações/mistura nos User-Agents')
    parser.add_argument('--show-headers', choices=['none','mask','full'], default='mask', help='Mostrar headers antes da request (mask oculta token parcialmente)')
    parser.add_argument('--show-response-headers', choices=['none','mask','full'], default='none', help='Mostrar headers da resposta (none/mask/full)')
    args = parser.parse_args()

    urls = [u for u in API_URL if u]
    if not urls:
        print('Erro: lista de URLs vazia.')
        raise SystemExit(1)

    # carrega tokens do ficheiro e adiciona fallback (BEARER_TOKEN está intencionalmente vazio)
    tokens = load_tokens_from_file(args.tokens_file)
    tokens += BEARER_TOKEN

    if not tokens:
        print('Aviso: nenhum token encontrado, a usar tokens fake para testes.')
        tokens = [generate_fake_jwt() for _ in range(5)]

    for i in range(args.count):
        print(f"\n--- Request {i+1} ---")
        data = make_request(urls, tokens, USER_AGENTS, mix_ua=args.mix_ua, show_headers=args.show_headers, show_response_headers=args.show_response_headers)
        print(data)
        time.sleep(random.uniform(args.min_delay, args.max_delay))

