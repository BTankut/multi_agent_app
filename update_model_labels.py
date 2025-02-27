#!/usr/bin/env python3
"""
Model Etiketleri Güncelleme Aracı
OpenRouter API'den güncel model listesini çekerek model_labels.json dosyasını günceller.
Mevcut etiketleri korur ve yeni modeller için akıllı etiketleme yapar.
"""

import os
import json
import re
import requests
import logging
import time
from dotenv import load_dotenv

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Sabit dosya yolları
MODEL_LABELS_FILE = "data/model_labels.json"
MODEL_ROLES_FILE = "data/model_roles.json"
BACKUP_DIR = "data/backups"

def ensure_backup_dir():
    """Yedekleme dizininin var olduğundan emin olur"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"Yedekleme dizini oluşturuldu: {BACKUP_DIR}")

def load_json(file_path):
    """JSON dosyasını yükler"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Dosya yüklenirken hata: {e}")
        return None

def save_json(file_path, data):
    """JSON verilerini dosyaya kaydeder"""
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        logger.info(f"Dosya başarıyla kaydedildi: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Dosya kaydedilirken hata: {e}")
        return False

def backup_file(file_path):
    """Bir dosyanın zaman damgalı yedeğini oluşturur ve eski yedekleri temizler"""
    ensure_backup_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.basename(file_path)
    backup_path = f"{BACKUP_DIR}/{filename}-{timestamp}.bak"
    
    try:
        data = load_json(file_path)
        if data:
            save_json(backup_path, data)
            logger.info(f"Yedek oluşturuldu: {backup_path}")
            
            # Eski yedekleri temizle - her dosya türü için en son 5 yedek dışındakileri sil
            clean_old_backups(filename)
            return True
    except Exception as e:
        logger.error(f"Yedekleme sırasında hata: {e}")
    
    return False

def clean_old_backups(filename_prefix):
    """Belirli bir türdeki eski yedekleri temizler, sadece en son 5 tanesini tutar"""
    try:
        # İlgili dosya yedeklerini bul
        backup_files = []
        for file in os.listdir(BACKUP_DIR):
            if file.startswith(filename_prefix) and file.endswith(".bak"):
                file_path = os.path.join(BACKUP_DIR, file)
                backup_files.append((file_path, os.path.getmtime(file_path)))
        
        # Son değişiklik zamanına göre sırala (en yeni en sonda)
        backup_files.sort(key=lambda x: x[1])
        
        # En son 5 yedek dışındakileri sil
        if len(backup_files) > 5:
            for file_path, _ in backup_files[:-5]:  # En eski dosyaları sil
                os.remove(file_path)
                logger.info(f"Eski yedek temizlendi: {file_path}")
            
            logger.info(f"{len(backup_files)-5} eski yedek temizlendi, {min(5, len(backup_files))} yedek tutuldu")
    except Exception as e:
        logger.error(f"Eski yedekler temizlenirken hata: {e}")

def get_openrouter_models():
    """OpenRouter'dan mevcut modelleri çeker"""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY bulunamadı!")
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
        models = response.json().get('data', [])
        logger.info(f"OpenRouter'dan {len(models)} model alındı")
        return models
    except Exception as e:
        logger.error(f"OpenRouter'dan model listesi alınırken hata: {e}")
        return []

def get_available_labels():
    """model_roles.json'dan mevcut etiketleri yükler"""
    roles_data = load_json(MODEL_ROLES_FILE)
    if not roles_data or "labels" not in roles_data:
        logger.error("Model rolleri yüklenemedi veya 'labels' anahtarı bulunamadı")
        return []
    
    return [label_entry['label'] for label_entry in roles_data['labels']]

def get_existing_model_labels():
    """Mevcut model_labels.json'dan model-etiket eşleştirmelerini yükler"""
    model_data = load_json(MODEL_LABELS_FILE)
    if not model_data:
        logger.error("Model etiketleri yüklenemedi")
        return {}
    
    # Model ve etiketlerini sözlük olarak döndür
    return {entry.get("model", ""): entry.get("labels", []) for entry in model_data if "model" in entry}

def determine_labels_for_model(model_info, available_labels, existing_labels):
    """
    Bir model için etiketleri belirler. Önce mevcut etiketleri kullanır, 
    model yeni ise akıllı etiketleme yapar.
    """
    model_id = model_info.get('id', '')
    
    # Eğer model zaten etiketlenmişse, mevcut etiketleri kullan
    if model_id in existing_labels:
        return existing_labels[model_id]
    
    # Yeni model için etiketleri belirle
    labels = ["general_assistant"]  # Her model en az genel asistan olarak işaretlenir
    
    # Ücret durumuna göre etiketleme
    try:
        prompt_price = model_info.get('pricing', {}).get('prompt', '0')
        # Sayısal değere dönüştür (string olabilir)
        prompt_price = float(prompt_price) if prompt_price else 0
        
        if prompt_price > 0:
            labels.append("paid") 
        else:
            labels.append("free")
    except (ValueError, TypeError):
        # Dönüştürme hatası durumunda varsayılan olarak free etiketini ekle
        labels.append("free")
    
    # Model ID'sine göre otomatik etiketleme
    model_id_lower = model_id.lower()
    model_name = model_info.get('name', '').lower()
    
    # Code/Coding uzmanı modelleri
    if any(term in model_id_lower or term in model_name for term in ["code", "coding", "coder", "codestral", "phi-3"]):
        if "code_expert" in available_labels:
            labels.append("code_expert")
    
    # Matematik uzmanı modelleri
    if any(term in model_id_lower or term in model_name for term in ["math", "numeric", "gemini", "gpt-4", "claude-3", "o1", "mixtral"]):
        if "math_expert" in available_labels:
            labels.append("math_expert")
    
    # Görüntü işleme modelleri
    if any(term in model_id_lower or term in model_name for term in ["vision", "vl", "image", "pixtral", "visual"]):
        if "vision_expert" in available_labels:
            labels.append("vision_expert")
    
    # Deneysel modeller
    if any(term in model_id_lower for term in ["exp", "experimental", "beta", "preview"]):
        if "experimental" in available_labels:
            labels.append("experimental")
    
    # Akıl yürütme uzmanları
    if any(term in model_id_lower or term in model_name for term in ["reasoning", "gemini", "claude", "gpt-4", "o1", "mixtral"]):
        if "reasoning_expert" in available_labels:
            labels.append("reasoning_expert")
    
    # Hızlı yanıt modelleri
    if any(term in model_id_lower for term in ["flash", "haiku", "mini", "small", "fast"]):
        if "fast_response" in available_labels:
            labels.append("fast_response")
    
    # Talimat takip etme
    if "instruct" in model_id_lower:
        if "instruction_following" in available_labels:
            labels.append("instruction_following")
    
    # Çokdilli modeller
    if any(term in model_id_lower for term in ["multilingual", "multi-lingual"]):
        if "multilingual" in available_labels:
            labels.append("multilingual")
    
    # Model boyutuna göre etiketleme
    size_match = re.search(r'(\d+)[bB]', model_id)
    if size_match:
        size = int(size_match.group(1))
        if size >= 70:  # Büyük modeller genellikle akıl yürütmede daha iyidir
            if "reasoning_expert" in available_labels and "reasoning_expert" not in labels:
                labels.append("reasoning_expert")
    
    # Etiketleri benzersiz yap
    return list(set(labels))

def update_model_labels():
    """
    OpenRouter API'den model listesini çeker ve model_labels.json dosyasını günceller.
    Mevcut etiketleri korur ve yeni modeller için otomatik etiketleme yapar.
    """
    # Mevcut dosyaları yedekle
    if not backup_file(MODEL_LABELS_FILE):
        # Web arayüzünde çalıştığında input sorun yaratabilir, bu yüzden yedekleme hatası olsa bile devam et
        logger.warning("Yedekleme yapılamadı, devam ediliyor...")
    
    # Veri kaynaklarını yükle
    openrouter_models = get_openrouter_models()
    available_labels = get_available_labels()
    existing_labels = get_existing_model_labels()
    
    if not openrouter_models:
        logger.error("OpenRouter modelleri alınamadı, işlem iptal ediliyor")
        return False
    
    if not available_labels:
        logger.error("Kullanılabilir etiketler yüklenemedi, işlem iptal ediliyor")
        return False
    
    # Yeni model_labels.json verisi oluştur
    new_model_labels = []
    updated_count = 0
    new_count = 0
    
    for model in openrouter_models:
        model_id = model.get('id', '')
        if not model_id:
            continue
        
        # Model etiketlerini belirle
        labels = determine_labels_for_model(model, available_labels, existing_labels)
        
        # Yeni veya güncellenen model sayısını izle
        if model_id in existing_labels:
            if set(labels) != set(existing_labels[model_id]):
                updated_count += 1
        else:
            new_count += 1
        
        # Modeli yeni listeye ekle
        new_model_labels.append({
            "model": model_id,
            "labels": labels
        })
    
    # Değişiklikleri kaydet
    if save_json(MODEL_LABELS_FILE, new_model_labels):
        logger.info(f"Model etiketleri güncellendi: {len(new_model_labels)} toplam model")
        logger.info(f"  - {new_count} yeni model eklendi")
        logger.info(f"  - {updated_count} mevcut model güncellendi")
        return True
    else:
        logger.error("Model etiketleri güncellenirken hata oluştu")
        return False

if __name__ == "__main__":
    logger.info("Model etiketleri güncelleme aracı başlatılıyor...")
    update_model_labels()