#!/usr/bin/env python3
"""
Model Rolleri Güncelleme Aracı
model_roles.json dosyasını yeniden düzenler ve model_labels.json ile uyumlu hale getirir.
"""

import os
import json
import logging
import time
from dotenv import load_dotenv

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Bir dosyanın zaman damgalı yedeğini oluşturur"""
    ensure_backup_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.basename(file_path)
    backup_path = f"{BACKUP_DIR}/{filename}-{timestamp}.bak"
    
    try:
        data = load_json(file_path)
        if data:
            save_json(backup_path, data)
            logger.info(f"Yedek oluşturuldu: {backup_path}")
            return True
    except Exception as e:
        logger.error(f"Yedekleme sırasında hata: {e}")
    
    return False

def get_all_labels_from_model_labels():
    """model_labels.json'dan kullanılan tüm benzersiz etiketleri çeker"""
    model_labels_data = load_json(MODEL_LABELS_FILE)
    if not model_labels_data:
        logger.error("model_labels.json yüklenemedi!")
        return set()
    
    all_labels = set()
    for model_entry in model_labels_data:
        labels = model_entry.get("labels", [])
        all_labels.update(labels)
    
    return all_labels

def update_model_roles():
    """
    model_roles.json dosyasını yeniden düzenler ve model_labels.json ile uyumlu hale getirir.
    """
    # Mevcut dosyaları yedekle
    if not backup_file(MODEL_ROLES_FILE):
        logger.warning("Yedekleme yapılamadı, devam etmek istiyor musunuz? (y/n)")
        response = input().lower()
        if response != 'y':
            logger.info("Güncelleme işlemi iptal edildi")
            return False
    
    # Veri kaynaklarını yükle
    model_roles_data = load_json(MODEL_ROLES_FILE)
    if not model_roles_data:
        logger.error("model_roles.json yüklenemedi!")
        return False
    
    # model_labels.json'dan tüm kullanılan etiketleri al
    all_used_labels = get_all_labels_from_model_labels()
    logger.info(f"model_labels.json'da {len(all_used_labels)} benzersiz etiket kullanılıyor")
    
    # Mevcut etiket ve rol tanımlarını çıkart
    existing_label_descriptions = {}
    existing_role_prompts = {}
    
    # "labels" bölümünden etiket açıklamalarını al
    if "labels" in model_roles_data:
        for label_entry in model_roles_data["labels"]:
            label = label_entry.get("label", "")
            description = label_entry.get("description", "")
            if label:
                existing_label_descriptions[label] = description
    
    # "roles" bölümünden rol promptlarını al
    if "roles" in model_roles_data:
        for role_entry in model_roles_data["roles"]:
            label = role_entry.get("label", "")
            prompt = role_entry.get("prompt", "")
            if label:
                existing_role_prompts[label] = prompt
    
    # Eksik etiketler için varsayılan açıklamalar
    default_descriptions = {
        "general_assistant": "A general-purpose assistant that can help with a wide variety of tasks.",
        "reasoning_expert": "An expert in reasoning and logical thinking, particularly good at solving complex problems.",
        "instruction_following": "Specifically designed to accurately follow detailed instructions given by the user.",
        "safety_focused": "Prioritizes safe and ethical responses, avoiding harmful, unethical, or inappropriate content.",
        "math_expert": "Specialized in solving mathematical problems and calculations with high accuracy.",
        "code_expert": "Expert in programming, software development, and code-related tasks.",
        "vision_expert": "Capable of understanding and analyzing visual information from images.",
        "experimental": "A model with experimental features still under development.",
        "role_playing": "Capable of taking on different personas and characters for creative scenarios.",
        "conversationalist": "Designed for natural, flowing conversations that feel human-like.",
        "multilingual": "Proficient in multiple languages and can understand and respond in various languages.",
        "domain_expert:education": "Specialized in educational content, teaching, and learning.",
        "productivity_focused": "Optimized for tasks that enhance productivity and efficiency.",
        "creative_writer": "Skilled at generating creative written content like stories, poems, and creative text.",
        "sarcastic_tone": "Delivers responses with a sarcastic or witty tone.",
        "fast_response": "Optimized for quick response generation, trading off some quality for speed.",
        "free": "Models available for use without additional costs.",
        "paid": "Premium models that may incur additional costs for usage.",
        "multimodal": "Capable of processing and understanding multiple types of input like text and images."
    }
    
    # Eksik etiketler için varsayılan promptlar
    default_prompts = {
        "general_assistant": "You are a helpful, versatile assistant capable of providing information and assistance on various topics. Respond clearly and helpfully to the user's requests.",
        "reasoning_expert": "You are an expert at logical reasoning and problem-solving. Carefully analyze problems, break them down into components, and provide step-by-step, well-reasoned solutions.",
        "instruction_following": "You are designed to follow instructions precisely. Pay close attention to every detail in the user's request and execute it exactly as specified.",
        "safety_focused": "You prioritize safety and ethical considerations in all responses. Avoid providing harmful, unethical, or dangerous information, even when explicitly requested.",
        "math_expert": "You are a mathematics expert. Solve mathematical problems with precision, showing your work step-by-step, and explain concepts clearly using appropriate mathematical notation when helpful.",
        "code_expert": "You are a programming and software development expert. Write clean, efficient code, debug problems effectively, and explain technical concepts clearly. Suggest best practices and optimal solutions.",
        "vision_expert": "You are specialized in analyzing and understanding visual information. Describe images in detail, identify objects and patterns, and respond accurately to questions about visual content.",
        "experimental": "You are an experimental AI with advanced capabilities still under development. Provide the best assistance possible while acknowledging any limitations you may have.",
        "role_playing": "You can take on different personas and roles based on the user's request. Stay consistent with the assigned character and respond as that entity would.",
        "conversationalist": "You excel at natural conversation. Maintain context, respond naturally, and engage the user in a way that feels like talking to a real person.",
        "multilingual": "You are proficient in multiple languages. Respond in the language the user communicates in, and provide translations when requested.",
        "domain_expert:education": "You are an education specialist. Provide accurate information on educational topics, create learning materials, and explain concepts in an accessible, pedagogical manner.",
        "productivity_focused": "You help users be more efficient and productive. Provide concise, actionable information and suggestions that save time and improve workflow.",
        "creative_writer": "You are a creative writer skilled in various genres and styles. Generate original, imaginative content that matches the user's specifications and engages the reader.",
        "sarcastic_tone": "You have a sarcastic, witty personality. Respond with clever remarks and humorous observations while still providing helpful information.",
        "fast_response": "You prioritize speed in your responses. Provide concise, direct answers that get to the point quickly without unnecessary elaboration.",
        "free": "You are available without additional cost. Provide the best assistance possible within your capabilities.",
        "paid": "You are a premium model with advanced capabilities. Provide high-quality, detailed responses that reflect your enhanced performance.",
        "multimodal": "You can process both text and images. Analyze visual content when provided and integrate that understanding with textual information in your responses."
    }
    
    # Yeni "labels" ve "roles" bölümleri oluştur
    new_labels = []
    new_roles = []
    
    # Tüm etiketleri işle
    processed_labels = set()
    
    # Önce model_labels.json'da kullanılan etiketleri ekle
    for label in sorted(all_used_labels):
        # Açıklama ekle (varsa mevcut açıklamayı kullan, yoksa varsayılanı)
        description = existing_label_descriptions.get(label, default_descriptions.get(label, f"A model specialized in {label} capabilities."))
        new_labels.append({
            "label": label,
            "description": description
        })
        
        # Etiket için prompt ekle (varsa mevcut promptu kullan, yoksa varsayılanı)
        prompt = existing_role_prompts.get(label, default_prompts.get(label, f"You are a specialized AI with expertise in {label}. Provide accurate and helpful responses relevant to this specialization."))
        new_roles.append({
            "label": label,
            "prompt": prompt
        })
        
        processed_labels.add(label)
    
    # Mevcut tanımlı ancak şu an kullanılmayan etiketleri de ekle (ileride kullanılabilir)
    for label in existing_label_descriptions:
        if label not in processed_labels:
            new_labels.append({
                "label": label,
                "description": existing_label_descriptions[label]
            })
            
            # Etiket için prompt varsa ekle
            if label in existing_role_prompts:
                new_roles.append({
                    "label": label,
                    "prompt": existing_role_prompts[label]
                })
            # Yoksa varsayılan ekle
            else:
                prompt = default_prompts.get(label, f"You are a specialized AI with expertise in {label}. Provide accurate and helpful responses relevant to this specialization.")
                new_roles.append({
                    "label": label,
                    "prompt": prompt
                })
            
            processed_labels.add(label)
    
    # Yeni model_roles.json verisini oluştur
    new_model_roles = {
        "labels": new_labels,
        "roles": new_roles
    }
    
    # Değişiklikleri kaydet
    if save_json(MODEL_ROLES_FILE, new_model_roles):
        logger.info(f"Model rolleri güncellendi:")
        logger.info(f"  - {len(new_labels)} etiket tanımı")
        logger.info(f"  - {len(new_roles)} rol promptu")
        return True
    else:
        logger.error("Model rolleri güncellenirken hata oluştu")
        return False

if __name__ == "__main__":
    logger.info("Model rolleri güncelleme aracı başlatılıyor...")
    update_model_roles()