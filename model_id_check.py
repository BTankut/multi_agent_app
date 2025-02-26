#!/usr/bin/env python3
"""
Model ID Uyumsuzluk Kontrol Aracı
Bu script model_labels.json'daki modellerin OpenRouter'daki gerçek model ID'leriyle uyumluluğunu kontrol eder.
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

def check_model_id_compatibility():
    """model_labels.json'daki model ID'lerini OpenRouter modelleriyle karşılaştırır"""
    # Dosyadan model etiketlerini yükle
    model_labels_data = load_json("data/model_labels.json")
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
    
    # Uyumluluğu kontrol et
    print("\n===== MODEL ID UYUMLULUK KONTROLÜ =====")
    print(f"OpenRouter'da toplam {len(openrouter_model_ids)} model bulundu")
    print(f"model_labels.json'da toplam {len(model_labels_data)} model tanımı var")
    
    missing_models = []
    matching_models = []
    
    for model_entry in model_labels_data:
        model_id = model_entry.get("model", "")
        if model_id in openrouter_model_ids:
            matching_models.append(model_id)
        else:
            missing_models.append(model_id)
    
    # Sonuçları göster
    print(f"\n✅ OpenRouter'da bulunan modeller: {len(matching_models)}")
    for model in matching_models[:10]:  # İlk 10 modeli göster
        print(f"  - {model}")
    if len(matching_models) > 10:
        print(f"  ... ve {len(matching_models) - 10} model daha")
    
    print(f"\n❌ OpenRouter'da bulunamayan modeller: {len(missing_models)}")
    for model in missing_models[:30]:  # İlk 30 eksik modeli göster
        print(f"  - {model}")
    if len(missing_models) > 30:
        print(f"  ... ve {len(missing_models) - 30} model daha")
    
    # Formatları karşılaştır
    print("\n===== MODEL ID FORMAT KARŞILAŞTIRMASI =====")
    if openrouter_model_ids:
        print("OpenRouter Model ID örneği:")
        for model_id in openrouter_model_ids[:5]:
            print(f"  - {model_id}")
    
    # Adlandırma düzeltme önerileri oluştur
    possible_matches = find_possible_matches(missing_models, openrouter_model_ids)
    if possible_matches:
        print("\n===== OLASI EŞLEŞMELER =====")
        print("Eksik modellerinizin OpenRouter'da karşılığı olabilecek ID'ler:")
        for old_id, potential_matches in possible_matches.items():
            if potential_matches:
                print(f"\n{old_id} → Olası eşleşmeler:")
                for match in potential_matches[:3]:  # En iyi 3 eşleşmeyi göster
                    print(f"  - {match}")

def find_possible_matches(missing_models, openrouter_model_ids):
    """Eksik modeller için olası eşleşmeler önerir"""
    possible_matches = {}
    
    for missing_model in missing_models:
        # Model adının parçalarını al
        parts = missing_model.split(':')[0].split('/')
        if len(parts) == 1:  # Format: "EVA Qwen2.5 32B" gibi
            model_name = parts[0].lower()
            matches = []
            for or_id in openrouter_model_ids:
                # Model adının parçalarını ara
                words = model_name.split()
                match_score = sum(1 for word in words if word.lower() in or_id.lower())
                if match_score > 0:
                    matches.append((or_id, match_score))
            
            # Skor göre sırala
            matches.sort(key=lambda x: x[1], reverse=True)
            possible_matches[missing_model] = [m[0] for m in matches[:3]] if matches else []
        
        elif len(parts) == 2:  # Format: "google/gemini" gibi
            provider, model_name = parts
            matches = []
            for or_id in openrouter_model_ids:
                if provider.lower() in or_id.lower() and model_name.lower() in or_id.lower():
                    matches.append(or_id)
            possible_matches[missing_model] = matches
    
    return possible_matches

if __name__ == "__main__":
    check_model_id_compatibility()