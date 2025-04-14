import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from dotenv import load_dotenv
from contextlib import nullcontext
import re
import json
load_dotenv()

class LLM:
    def __init__(self, llm_model='gemini-2.0-flash', use_custom_llm=False, custom_llm_model='llama2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_custom_llm = use_custom_llm
        self.llm = None
        self.tokenizer = None
        self.custom_llm_model = custom_llm_model
        print(f"Using device: {self.device}")

        # Define model paths for the available options
        self.model_options = {
            'llama2': 'meta-llama/Llama-2-7b-chat-hf',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.1'
        }

        if self.use_custom_llm:
            try:
                # Check if the custom_llm_model is one of our predefined options
                if custom_llm_model in self.model_options:
                    model_path = self.model_options[custom_llm_model]
                    print(f"Loading {custom_llm_model} model from {model_path}")
                    self._init_custom_llm(model_path)
                else:
                    # Assume it's a direct model path
                    print(f"Loading custom model from path: {custom_llm_model}")
                    self._init_custom_llm(custom_llm_model)
            except Exception as e:
                print(f"Custom LLM failed to load: {e}, falling back to Gemini")
                self.use_custom_llm = False
                self._init_gemini_api(llm_model)
        else:
            self._init_gemini_api(llm_model)

    def _init_custom_llm(self, model_name):
        print(f"Loading custom LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add trust_remote_code if needed for some models
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            load_in_8bit=True if torch.cuda.is_available() else False,
            trust_remote_code=True  # May be needed for some models
        )
        
        # Ensure model is in eval mode
        self.llm.eval()
        print(f"Model parameters: {sum(p.numel() for p in self.llm.parameters())/1e6:.2f}M")
        print(f"Custom LLM loaded successfully: {model_name}")

    def _init_gemini_api(self, model_name):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        self.gemini_model_name = model_name
        print(f"Initialized Gemini API with model: {model_name}")

    def _format_context(self, contexts):
        """Helper method to format contexts consistently"""
        if isinstance(contexts, dict):
            # Original dictionary format
            formatted_context = "\n".join(
                f"Table: {tbl}\nColumns & Description: {desc}" for tbl, desc in contexts.items()
            )
        elif isinstance(contexts, list):
            # List of dictionaries format
            formatted_context = ""
            for context_dict in contexts:
                for tbl, desc in context_dict.items():
                    formatted_context += f"Table: {tbl}\nColumns & Description: {desc}\n\n"
        else:
            raise TypeError("contexts must be either a dictionary or a list of dictionaries")
        return formatted_context

    def get_answer(self, contexts, question: str, is_analytics=False):
        """
        Get answer from LLM based on the context and question
        
        Args:
            contexts: Dictionary or list of dictionaries containing table information
            question: Question to ask the LLM
            is_analytics: Boolean flag to determine if analytics prompt should be used
        """
        formatted_context = self._format_context(contexts)
        
        if is_analytics:
            prompt = self._create_analytics_prompt(formatted_context, question)
        else:
            prompt = self._create_standard_prompt(formatted_context, question)

        try:
            if self.use_custom_llm:
                return self._generate_with_custom_llm(prompt)
            else:
                return self._generate_with_gemini(prompt)
        except Exception as final_error:
            print(f"All models failed with error: {final_error}")
            return "Sorry, an error occurred while trying to generate a response."

    def _create_standard_prompt(self, formatted_context, question):
        """Creates the standard prompt for question answering"""
        return f"""You are an AI assistant trained to answer questions about hotel booking datasets.

        Use the context information below to answer the user's question precisely and return your response in JSON format.

        When analyzing the data:
        1. Calculate metrics with mathematical precision
        2. Include proper units for all measurements (e.g., '$' for revenue/price, '%' for percentages)
        3. Show your step-by-step calculation process in the analysis section
        4. Provide clear insights based on the data patterns
        5. Return ONLY valid, properly formatted JSON that can be parsed programmatically

        Context:
        {formatted_context}

        Question:
        {question}

        Your response must strictly follow this JSON structure:
        {{
        "result": "The direct answer to the question with appropriate units",
        "analysis": "Step-by-step calculation and reasoning process that led to this result",
        "insights": "Additional observations or patterns noticed in the data that might be relevant"
        }}
        """

    def _create_analytics_prompt(self, formatted_context, query):
        """Creates the analytics prompt for table analysis"""
        return f"""You are an AI analytics expert specializing in hotel data analysis.

        TASK: Analyze the available tables and determine which one is most relevant for answering the analytics query.
        Then provide a comprehensive analysis of that table in relation to the query.

        Available Tables:
        {formatted_context}

        Analytics Query:
        {query}

        Follow these steps:
        1. First, identify which table is most relevant to the analytics query
        2. Explain why you selected this table (structure, fields, data relevance)
        3. Provide a detailed analysis of what insights can be derived from this table
        4. Suggest potential data visualizations that would be valuable
        5. Note any limitations of this table for the requested analysis

        Your response must strictly follow this JSON structure:
        {{
        "selected_table": "Name of the most relevant table",
        "selection_reasoning": "Why this table is the best match for the query",
        "key_insights": "Main insights that can be derived from this table for the query",
        "recommended_visualizations": "List of visualization types that would be helpful",
        "limitations": "Any limitations or additional data that would be beneficial",
        "suggested_queries": "2-3 follow-up queries that would provide further valuable insights"
        }}

        Ensure your response is properly formatted JSON that can be programmatically parsed.
        """

    def _generate_with_custom_llm(self, prompt):
        try:
            # For LLaMa 2 models, prepare the prompt with proper chat template
            if "llama-2" in self.custom_llm_model.lower():
                system_part = "You are an AI assistant trained to answer questions about hotel booking datasets."
                user_part = prompt.replace(system_part, "").strip()
                
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_part}\n<</SYS>>\n\n{user_part} [/INST]"
                prompt = formatted_prompt
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                with torch.no_grad():
                    # Only get new tokens, not the input
                    input_length = inputs["input_ids"].shape[1]
                    outputs = self.llm.generate(
                        inputs["input_ids"],
                        max_new_tokens=512,
                        temperature=0.7,
                        num_return_sequences=1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    json.loads(json_str)  
                    return json_str
                except json.JSONDecodeError:
                    pass
            return response
        except Exception as e:
            raise RuntimeError(f"Custom LLM generation failed: {e}")

    def _generate_with_gemini(self, prompt):
        try:
            generation_config = genai.GenerationConfig(
                temperature=0.6,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,  # Increased token limit for analytics responses
            )
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if hasattr(response, 'text'):
                result_text = response.text
            else:
                # Handle newer API response format
                result_text = response.parts[0].text if hasattr(response, 'parts') else str(response)
                
            # Extract JSON if present
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                try:
                    # Try to parse as JSON to validate
                    json_str = json_match.group(1)
                    json.loads(json_str)  # Test if valid JSON
                    return json_str
                except json.JSONDecodeError:
                    pass
            # Return full response if no valid JSON found
            return result_text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")