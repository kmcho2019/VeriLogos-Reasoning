import requests
import json

class R1APIClient:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.base_url = self._get_base_url()

    def _get_base_url(self):
        if self.provider == "together":
            return "https://api.together.xyz/v1/completions" # Example, check actual endpoint
        elif self.provider == "fireworks":
            return "https://api.fireworks.ai/inference/v1/completions" # Example, check actual endpoint
        elif self.provider == "deepseek_api":
            return "https://api.deepseek.com/v1/chat/completions" # Example, check actual endpoint
        else:
            raise ValueError(f"Unsupported R1 API provider: {self.provider}")

    def generate_trace(self, masked_code: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") -> str:
        """
        Generates a reasoning trace and completion for the masked Verilog code.
        Adjust payload and headers based on the specific API provider's documentation.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Example payload - THIS WILL VARY SIGNIFICANTLY BETWEEN PROVIDERS
        # You'll need to consult the documentation for Together, Fireworks, or DeepSeek's R1 API
        payload = {}
        if self.provider == "deepseek_api": # DeepSeek has a chat completions like API
             payload = {
                "model": model_name, # Or the specific model served by the API
                "messages": [
                    {"role": "system", "content": "You are a Verilog expert. Complete the following masked Verilog code, providing the reasoning steps if possible."},
                    {"role": "user", "content": f"```verilog\n{masked_code}\n```\nComplete the [BLANK] sections."}
                ],
                "max_tokens": 1024, # Adjust as needed
                # "temperature": 0.7, # Adjust as needed
            }
        else: # Other providers might use a simpler prompt-based completion
            payload = {
                "model": model_name, # This might be an alias used by the provider
                "prompt": f"Complete the [BLANK] sections in the following Verilog code. Show your reasoning steps if possible.\n\n```verilog\n{masked_code}\n```\n\nReasoning and Completion:",
                "max_tokens": 1024,
                # "temperature": 0.7,
                # ... other provider-specific parameters
            }

        print(f"Sending request to {self.provider} with model {model_name}")
        # print(f"Payload: {json.dumps(payload, indent=2)}") # For debugging

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            response_data = response.json()
            # print(f"Response data: {json.dumps(response_data, indent=2)}") # For debugging

            # Extracting the generated text - THIS IS HIGHLY PROVIDER-SPECIFIC
            if self.provider == "deepseek_api":
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    return response_data["choices"][0].get("message", {}).get("content", "")
            elif response_data.get("choices") and len(response_data["choices"]) > 0:
                 # Common pattern for many completion APIs
                return response_data["choices"][0].get("text", "") 
            else:
                print(f"Warning: Could not extract text from {self.provider} response: {response_data}")
                return "" # Or raise an error

        except requests.exceptions.RequestException as e:
            print(f"API request failed for {self.provider}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return "" # Or raise an error
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response from {self.provider}: {response.text}")
            return ""

if __name__ == '__main__':
    # Quick test - replace with your actual key and provider
    # client = R1APIClient(provider="deepseek_api", api_key="YOUR_DEEPSEEK_API_KEY")
    # masked_verilog = "module test(input clk, output reg q);\n  // [BLANK] \nendmodule"
    # trace = client.generate_trace(masked_verilog, model_name="deepseek-chat") # Use appropriate model for deepseek API
    # print("\nGenerated Trace:\n", trace)

    # client = R1APIClient(provider="together", api_key="YOUR_TOGETHER_API_KEY")
    # masked_verilog = "module test(input clk, output reg q);\n  // [BLANK] \nendmodule"
    # trace = client.generate_trace(masked_verilog, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") # Check model availability on Together
    # print("\nGenerated Trace:\n", trace)