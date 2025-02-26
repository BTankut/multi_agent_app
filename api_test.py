#!/usr/bin/env python3
"""
Hızlı API Test Dosyası
Bu script, spesifik modellerin OpenRouter API üzerinden erişilebilirliğini test eder.
Özellikle "choices not found" hatasını incelemek için tasarlanmıştır.
"""

import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("HATA: OPENROUTER_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın.")
    exit(1)

def test_model(model_name):
    """Test a specific OpenRouter model with a simple query"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello, give me a one-word response."}
    ]
    
    print(f"\n🧪 {model_name} modeli test ediliyor...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": messages
            },
            timeout=45
        )
        
        # Check if the request was successful
        response.raise_for_status()
        result = response.json()
        
        # Print the full response for debugging
        print("API Cevabı:")
        print(json.dumps(result, indent=2))
        
        # Check if 'choices' exists in the response
        if 'choices' not in result or not result['choices']:
            print(f"❌ HATA: '{model_name}' - 'choices' alanı bulunamadı veya boş.")
            return False
        
        # Extract the content
        content = result['choices'][0]['message']['content']
        print(f"✅ BAŞARILI: '{model_name}' - Cevap: {content}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ HATA: '{model_name}' - API isteği başarısız: {str(e)}")
        return False
    except KeyError as e:
        print(f"❌ HATA: '{model_name}' - Cevap format hatası: {str(e)}")
        print("Tam cevap:", response.text if 'response' in locals() else "Yok")
        return False
    except Exception as e:
        print(f"❌ HATA: '{model_name}' - Beklenmeyen hata: {str(e)}")
        return False

def main():
    """Main function to test multiple models"""
    # Hata aldığımız modeller
    problem_models = [
        "google/gemini-2.0-pro-exp-02-05:free", 
        "openai/o1"
    ]
    
    # Alternatif olarak kullanılabilecek modeller
    alternative_models = [
        "qwen/qwen-max",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "mistralai/mistral-large-2411"
    ]
    
    # Problem olan modelleri test et
    print("\n===== PROBLEM MODELLER =====")
    for model in problem_models:
        test_model(model)
        time.sleep(2)  # API rate limit'e yakalanmamak için bekle
    
    # Alternatif modelleri test et
    print("\n===== ALTERNATİF MODELLER =====")
    for model in alternative_models:
        test_model(model)
        time.sleep(2)  # API rate limit'e yakalanmamak için bekle

if __name__ == "__main__":
    main()