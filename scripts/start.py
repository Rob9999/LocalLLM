import argparse
import os

# from local_LLM.gpt_model_wrapper import GPTModelWrapper
from local_LLM.gpt_model_wrapper import GPTModelWrapper
from local_LLM.webserver.webserver import LLMWebServer


def main():
    parser = argparse.ArgumentParser(description="Start Local LLM")
    args = parser.parse_args()

    os.system("docker-compose up -d")
    web_server = LLMWebServer(GPTModelWrapper())
    web_server.start_web_server()


if __name__ == "__main__":
    main()
