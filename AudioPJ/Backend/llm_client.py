import lmstudio as lms
from typing import Optional

class LLMStudioClient:
    def __init__(self, model_path: str = "gpt-oss-20b"):
        self.model_path = model_path

    def generate_lyrics(self, prompt: str, genre: str) -> str:
        """
        Returns fixed lyrics to save resources.
        """
        return """VERSO
Ali spezzate nel vento gridano
Fuoco e sangue sulle montagne cadono
Un urlo muto che nessuno sente
Un dio tradito dalla sua stessa gente

PRE
Sofferenza scolpita nel marmo
Paura dipinta nell'ombra

CORO
Esilio eterno tra gli dèi e i titani
Rabbia che brucia e divora le mani
Lacrime del cielo che cadono giù
Nel buio profondo non tornerai più

VERSO 2
Lame di luce tagliano il silenzio
Un angelo piange nel suo dissenso
Catene invisibili stringono l'anima
Il dolore divora come fiamma viva

CORO
Esilio eterno tra gli dèi e i titani
Rabbia che brucia e divora le mani
Lacrime del cielo che cadono giù
Nel buio profondo non tornerai più

BRIDGE
Tradimento inciso sulla pelle
Il tempo si ferma tra queste stelle
Siamo soli nell’eco del nulla
Siamo ombre senza culla"""

    def generate_music(self, prompt: str, genre: str) -> str:
        """
        Generates music using YuE model in LM Studio.
        Note: This assumes YuE is loaded and returns audio tokens or a path.
        """
        try:
            # Use context manager to ensure fresh connection
            with lms.Client() as client:
                # Dynamic discovery of YuE model using requests (SDK list() is broken)
                model_path_to_use = self.model_path
                import requests
                try:
                    response = requests.get("http://localhost:1234/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        for m in data['data']:
                            if "yue" in m['id'].lower():
                                model_path_to_use = m['id']
                                print(f"DEBUG: Found YuE model: {model_path_to_use}")
                                break
                except Exception as e:
                    print(f"DEBUG: Could not list models via HTTP: {e}")

                print(f"DEBUG: Sending request to LM Studio model: {model_path_to_use}")
                
                # We might need to adjust the system prompt for YuE
                system_prompt = f"Generate a {genre} song based on this description: {prompt}"
                
                model = client.llm.model(model_path_to_use)
                
                print(f"DEBUG: System prompt: {system_prompt}")
                result = model.respond(system_prompt)
                print("DEBUG: Received response from LM Studio")
                
                # Clean Output (Remove CoT tags if present)
                content = result.content
                if "<|channel|>" in content:
                    # Remove everything before the last <|message|> or just strip the tags
                    # Heuristic: split by <|message|> and take the last part
                    parts = content.split("<|message|>")
                    if len(parts) > 1:
                        content = parts[-1]
                
                return content
        except Exception as e:
            print(f"Error generating music with LM Studio: {e}")
            return f"[Error: {e}]"

    def refine_prompt(self, raw_prompt: str) -> str:
        """
        Refines a user prompt to be more suitable for music generation models.
        """
        try:
            # Use context manager to ensure fresh connection and proper cleanup
            with lms.Client() as client:
                model = client.llm.model(self.model_path)
                system_prompt = "You are an expert prompt engineer for AI music generation. Refine the following user description into a detailed, comma-separated list of musical tags and descriptors (instruments, mood, tempo, era). Output ONLY the tags."
                
                result = model.respond(f"User description: {raw_prompt}")
                return result.content
        except Exception as e:
            print(f"Error refining prompt with LM Studio: {e}")
            return raw_prompt
