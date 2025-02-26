#!/usr/bin/env python3
"""
Model ID Düzeltme Aracı
Bu script model_labels.json'daki modellerin OpenRouter'daki gerçek model ID'leriyle uyumluluğunu sağlar.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def load_json(file_path):
    """JSON dosyasını yükler"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Dosya yüklenirken hata: {e}")
        return None

def save_json(file_path, data):
    """JSON verilerini dosyaya kaydeder"""
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"✅ Dosya başarıyla kaydedildi: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Dosya kaydedilirken hata: {e}")
        return False

def get_openrouter_models():
    """OpenRouter'dan mevcut modelleri çeker"""
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY bulunamadı!")
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=15
        )
        
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"OpenRouter'dan model listesi alınırken hata: {e}")
        return []

# Model ID düzeltme eşleşmeleri - Zamanla güncellenebilir
MODEL_ID_CORRECTIONS = {
    # Kullanılamayan modellerin doğru OpenRouter formatları
    "EVA Qwen2.5 32B": "eva-unit-01/eva-qwen-2.5-32b",
    "Unslopnemo 12B": "thedrummer/unslopnemo-12b",
    "Anthropic: Claude 3.5 Haiku (2024-10-22) (self-moderated)": "anthropic/claude-3.5-haiku",
    "Anthropic: Claude 3.5 Haiku (2024-10-22)": "anthropic/claude-3.5-haiku",
    "Anthropic: Claude 3.5 Haiku (self-moderated)": "anthropic/claude-3.5-haiku",
    "NeverSleep: Lumimaid v0.2 70B": "neversleep/llama-3-lumimaid-70b",
    "Magnum v4 72B": "anthracite-org/magnum-v4-72b",
    "Anthropic: Claude 3.5 Sonnet (self-moderated)": "anthropic/claude-3.5-sonnet",
    "xAI: Grok Beta": "x-ai/grok-2-1212",
    "Mistral: Ministral 8B": "mistralai/mistral-small-24b-instruct-2501:free",
    "Mistral: Ministral 3B": "mistralai/mistral-small-24b-instruct-2501:free",
    "Qwen2.5 7B Instruct": "qwen/qwen-2.5-7b-instruct",
    "NVIDIA: Llama 3.1 Nemotron 70B Instruct (free)": "nvidia/llama-3.1-nemotron-70b-instruct",
    "Magnum v2 72B": "anthracite-org/magnum-v2-72b",
    "Liquid: LFM 40B MoE": "liquid/lfm-7b",
    "Rocinante 12B": "thedrummer/rocinante-12b",
    "Meta: Llama 3.2 3B Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Meta: Llama 3.2 1B Instruct (free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Meta: Llama 3.2 1B Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Meta: Llama 3.2 90B Vision Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Meta: Llama 3.2 11B Vision Instruct (free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Meta: Llama 3.2 11B Vision Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Qwen2.5 72B Instruct": "qwen/qwen-2.5-72b-instruct",
    "Qwen2-VL 72B Instruct": "qwen/qwen-2-vl-72b-instruct",
    "NeverSleep: Lumimaid v0.2 8B": "neversleep/llama-3.1-lumimaid-8b",
    "OpenAI: o1-mini (2024-09-12)": "openai/o1-mini",
    "OpenAI: o1-preview (2024-09-12)": "openai/o1-preview",
    "Mistral: Pixtral 12B": "mistralai/pixtral-12b",
    "Cohere: Command R (08-2024)": "cohere/command-r-08-2024",
    "Cohere: Command R+ (08-2024)": "cohere/command-r-plus-08-2024"
}

def fix_model_ids():
    """model_labels.json'daki model ID'lerini düzeltir"""
    # Dosyadan model etiketlerini yükle
    file_path = "data/model_labels.json"
    model_labels_data = load_json(file_path)
    if not model_labels_data:
        print("model_labels.json yüklenemedi!")
        return
    
    # OpenRouter'dan mevcut modelleri çek
    openrouter_models = get_openrouter_models()
    if not openrouter_models:
        print("OpenRouter modellerine erişilemedi!")
        return
    
    # OpenRouter model ID'lerini listele
    openrouter_model_ids = [model.get('id', '') for model in openrouter_models]
    
    # Değişiklik sayacı
    changes_made = 0
    models_updated = []
    
    # Model ID'lerini düzelt
    for i, model_entry in enumerate(model_labels_data):
        old_model_id = model_entry.get("model", "")
        
        # ID'nin OpenRouter'da olup olmadığını kontrol et
        if old_model_id not in openrouter_model_ids:
            # Eşleştirme tablosunda varsa düzelt
            if old_model_id in MODEL_ID_CORRECTIONS:
                new_model_id = MODEL_ID_CORRECTIONS[old_model_id]
                # Düzeltilen ID'nin OpenRouter'da olduğunu teyit et
                if new_model_id in openrouter_model_ids:
                    model_labels_data[i]["model"] = new_model_id
                    changes_made += 1
                    models_updated.append((old_model_id, new_model_id))
                    print(f"✅ Model ID düzeltildi: {old_model_id} → {new_model_id}")
                else:
                    print(f"⚠️ Düzeltilen ID OpenRouter'da bulunamadı: {new_model_id}")
    
    # Değişiklikleri kaydet
    if changes_made > 0:
        print(f"\n{changes_made} model ID'si düzeltildi.")
        # Düzeltilen model ID'lerinin listesini göster
        print("\nDüzeltilen modeller:")
        for old_id, new_id in models_updated:
            print(f"  {old_id} → {new_id}")
        
        # Dosyayı kaydet
        backup_path = f"{file_path}.bak"
        # Önce yedek oluştur
        save_json(backup_path, load_json(file_path))
        print(f"Yedek dosya oluşturuldu: {backup_path}")
        
        # Değişiklikleri kaydet
        if save_json(file_path, model_labels_data):
            print("✅ Düzeltmeler başarıyla kaydedildi.")
        else:
            print("❌ Düzeltmeler kaydedilemedi.")
    else:
        print("Düzeltilecek model ID'si bulunamadı.")

if __name__ == "__main__":
    fix_model_ids()