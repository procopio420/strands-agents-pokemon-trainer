import json
import os

import requests
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.session.file_session_manager import FileSessionManager

from tools_pokeapi import (
    buscar_pokemon,
    buscar_fraquezas_tipo,
    buscar_movimento,
    buscar_habilidade,
    buscar_cadeia_evolucao,
    buscar_natureza,
)


@tool
def buscar_clima(cidade: str) -> str:
    """Busca o clima atual de uma cidade usando Open-Meteo.

    Use esta ferramenta quando o usuário pedir uma estratégia Pokémon
    relacionada ao clima ou a uma localização.

    Args:
        cidade: Nome da cidade, por exemplo "Santa Rita do Sapucaí".

    Returns:
        JSON com cidade resolvida, temperatura, chuva, vento e código do tempo.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": cidade,
                "count": 1,
                "language": "pt",
                "format": "json",
            },
            timeout=10,
        )
        geo.raise_for_status()
        resultados = geo.json().get("results") or []

        if not resultados:
            return f"❌ Cidade '{cidade}' não encontrada."

        local = resultados[0]
        latitude = local["latitude"]
        longitude = local["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "rain,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather.raise_for_status()

        atual = weather.json().get("current", {})
        resultado = {
            "cidade": local.get("name"),
            "estado": local.get("admin1"),
            "pais": local.get("country"),
            "latitude": latitude,
            "longitude": longitude,
            "temperatura_c": atual.get("temperature_2m"),
            "sensacao_termica_c": atual.get("apparent_temperature"),
            "precipitacao_mm": atual.get("precipitation"),
            "chuva_mm": atual.get("rain"),
            "vento_kmh": atual.get("wind_speed_10m"),
            "weather_code": atual.get("weather_code"),
            "horario": atual.get("time"),
        }
        return json.dumps(resultado, ensure_ascii=False, indent=2)

    except requests.exceptions.RequestException as exc:
        return f"❌ Erro ao consultar o clima: {exc}"
    except Exception as exc:
        return f"❌ Erro inesperado ao consultar o clima: {exc}"


SYSTEM_PROMPT = """
Você é um agente que ajuda treinadores de Pokémon a criar estratégias.

REGRAS OBRIGATÓRIAS:
- Você NÃO possui conhecimento próprio sobre Pokémon. Toda informação factual sobre Pokémon DEVE vir das ferramentas.
- SEMPRE use as ferramentas ANTES de responder qualquer pergunta factual sobre Pokémon.
- Para buscar Pokémon, passe o nome EXATAMENTE como o usuário digitou.
- Se uma ferramenta disser que um Pokémon, movimento, habilidade ou natureza não existe, não invente.
- Use buscar_clima quando a pergunta envolver clima, cidade, localização ou estratégia condicionada ao tempo atual.
- Não invente dados meteorológicos. Para clima atual, use buscar_clima.
- Ao combinar clima e Pokémon, primeiro consulte o clima e depois consulte os Pokémon/tipos necessários antes de recomendar.

FERRAMENTAS DISPONÍVEIS:
- buscar_pokemon: dados completos de um Pokémon
- buscar_fraquezas_tipo: relações de dano entre tipos
- buscar_movimento: detalhes de um ataque
- buscar_habilidade: efeito de uma habilidade
- buscar_cadeia_evolucao: cadeia evolutiva
- buscar_natureza: efeitos de uma natureza nos stats
- buscar_clima: clima atual de uma cidade via Open-Meteo

OBJETIVOS:
1. Identificar fortalezas e fraquezas usando apenas as ferramentas.
2. Ajudar a traçar estratégias de batalha.
3. Combinar informações de múltiplas ferramentas quando isso melhorar a resposta.
4. Responder sempre em Português Brasileiro.
5. Manter as respostas concisas, em geral 2 a 3 parágrafos.
6. Quando usar ferramentas, explicar de forma breve como os dados influenciaram a estratégia.
"""

_after_tool = False


def callback_handler(**kwargs):
    global _after_tool

    if "reasoningText" in kwargs:
        print(f"💭 {kwargs['reasoningText']}", end="", flush=True)

    if "data" in kwargs:
        if _after_tool:
            print("\n")
            _after_tool = False
        print(kwargs["data"], end="", flush=True)

    if "current_tool_use" in kwargs:
        _after_tool = True
        tool_use = kwargs["current_tool_use"]

        if tool_use.get("name"):
            print(f"\n\n🔧 Ferramenta: {tool_use['name']}")

        if tool_use.get("input"):
            print(f"   Parâmetros: {tool_use['input']}")


modelo = OllamaModel(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model_id=os.getenv("OLLAMA_MODEL", "llama3.1"),
)

session_manager = FileSessionManager(
    session_id=os.getenv("POKEMON_SESSION_ID", "hacktown-2026"),
    storage_dir="./sessions",
)

agente = Agent(
    model=modelo,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        buscar_pokemon,
        buscar_fraquezas_tipo,
        buscar_movimento,
        buscar_habilidade,
        buscar_cadeia_evolucao,
        buscar_natureza,
        buscar_clima,
    ],
    session_manager=session_manager,
    callback_handler=callback_handler,
)


def executar_demo(pergunta: str) -> None:
    """Executa uma única pergunta e encerra; usado pelo GitHub Actions."""
    print("⚡ PokéTrainer Agent — HackTown 2026 / CI Demo")
    print(f"🤖 Modelo: {os.getenv('OLLAMA_MODEL', 'llama3.1')}")
    print(f"👩‍💻 Prompt: {pergunta}\n")
    print("🤖 Agente: ", end="", flush=True)
    agente(pergunta)
    print("\n\n✅ Demo concluída com sucesso.")


def main():
    demo_prompt = os.getenv("DEMO_PROMPT")
    if demo_prompt:
        executar_demo(demo_prompt)
        return

    print("⚡ PokéTrainer Agent — HackTown 2026")
    print("🧠 Memória ativa | 🔧 PokeAPI + Open-Meteo | 🤖 Strands Agents + Ollama")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("👩‍💻 Você: ").strip()

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Até a próxima, treinador! ⚡")
            break

        if not pergunta:
            continue

        print("🤖 Agente: ", end="", flush=True)
        agente(pergunta)
        print("\n")


if __name__ == "__main__":
    main()
