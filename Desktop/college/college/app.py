import os
import pickle
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd
from flask import Flask, jsonify, request, render_template, session, redirect, url_for

from main import train_and_save_model


app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model_bundle.pkl")
SYMPTOM_ALIASES = {
    "fever": "fever",
    "headache": "headache",
    "head ache": "headache",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "fatigue": "fatigue",
    "joint pain": "joint_pain",
    "joint pains": "joint_pain",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "cough": "cough",
    "weight loss": "weight_loss",
    "weightloss": "weight_loss",
    "yellow eyes": "yellow_eyes",
    "yelloweye": "yellow_eyes",
}

# ===== Emergency Red-Flag Symptoms =====
EMERGENCY_SYMPTOMS = [
    "chest pain",
    "chest_pain",
    "difficulty breathing",
    "difficulty_breathing",
    "shortness of breath",
    "shortness_of_breath",
    "unconscious",
    "unconsciousness",
    "severe bleeding",
    "severe_bleeding",
    "severe chest pain",
    "severe_headache",
    "sudden numbness",
    "sudden_numbness",
    "slurred speech",
    "slurred_speech",
    "confusion",
    "seizure",
]

# ===== Disease Recommendations =====
DISEASE_RECOMMENDATIONS = {
    "Paralysis (brain hemorrhage)": {
        "home_care": [
            "Call emergency services immediately — this is a medical emergency",
            "Keep the patient calm and lying flat",
            "Loosen any tight clothing",
            "Do not give anything to eat or drink",
        ],
        "otc": ["Do not take any medication without doctor's advice"],
        "when_to_see_doctor": ["Seek immediate emergency medical attention"],
    },
    "Hypertension": {
        "home_care": [
            "Monitor blood pressure regularly",
            "Reduce salt intake in diet",
            "Maintain a healthy weight",
            "Exercise regularly (30 min/day)",
            "Limit alcohol consumption",
            "Quit smoking if applicable",
            "Practice stress management techniques",
        ],
        "otc": [
            "Over-the-counter pain relievers may affect blood pressure — consult a pharmacist",
            "Avoid decongestants containing phenylephrine",
        ],
        "when_to_see_doctor": [
            "If blood pressure readings are consistently above 140/90 mmHg",
            "If you experience severe headaches, vision changes, or chest pain",
            "For regular check-ups at least once every 3-6 months",
        ],
    },
    "Hepatitis B": {
        "home_care": [
            "Get plenty of rest",
            "Stay hydrated with water and clear fluids",
            "Avoid alcohol completely",
            "Eat small, frequent meals",
            "Avoid fatty or spicy foods",
        ],
        "otc": [
            "Acetaminophen (Tylenol) for fever — but do not exceed recommended dose",
            "Avoid ibuprofen and other NSAIDs without consulting a doctor",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of hepatitis (yellowing skin, dark urine, abdominal pain)",
            "For hepatitis B vaccination if not already vaccinated",
            "If you've been exposed to potentially infected blood or bodily fluids",
        ],
    },
    "Impetigo": {
        "home_care": [
            "Keep affected skin clean and dry",
            "Wash hands frequently",
            "Avoid scratching the sores",
            "Use separate towels and washcloths",
            "Keep children home from school until treatment starts",
        ],
        "otc": [
            "Antibacterial soap for gentle cleansing",
            "Topical antibiotic ointment (consult pharmacist for appropriate choice)",
        ],
        "when_to_see_doctor": [
            "If sores are spreading or worsening",
            "If you develop fever",
            "If sores are painful or appear infected",
            "Usually requires prescription antibiotic treatment",
        ],
    },
    "Chronic cholestasis": {
        "home_care": [
            "Follow a low-fat diet as advised by your doctor",
            "Take fat-soluble vitamin supplements as prescribed",
            "Avoid alcohol completely",
            "Stay well hydrated",
        ],
        "otc": [
            "Vitamin A, D, E, K supplements if recommended by doctor",
            "Anti-itch creams for pruritus (consult pharmacist)",
        ],
        "when_to_see_doctor": [
            "If you notice yellowing of skin or eyes",
            "If you have persistent itching",
            "For regular follow-up with a gastroenterologist",
        ],
    },
    "Hepatitis C": {
        "home_care": [
            "Get adequate rest",
            "Eat a balanced, nutritious diet",
            "Avoid alcohol completely",
            "Stay hydrated",
            "Avoid sharing personal items like razors or toothbrushes",
        ],
        "otc": [
            "Consult doctor before taking any new medications",
            "Avoid ibuprofen and NSAIDs",
            "Acetaminophen should be used cautiously and at low doses",
        ],
        "when_to_see_doctor": [
            "If you think you may have been exposed to hepatitis C",
            "For hepatitis C testing and treatment",
            "For regular liver function monitoring",
        ],
    },
    "Typhoid": {
        "home_care": [
            "Drink plenty of fluids (water, clear soups, oral rehydration solutions)",
            "Get plenty of rest",
            "Eat small, frequent, easily digestible meals",
            "Wash hands thoroughly and frequently",
            "Avoid preparing food for others",
        ],
        "otc": [
            "Acetaminophen for fever reduction",
            "Oral rehydration salts for dehydration prevention",
        ],
        "when_to_see_doctor": [
            "Typhoid requires antibiotic treatment — see a doctor immediately",
            "If you have prolonged fever, abdominal pain, or severe diarrhea",
            "If symptoms worsen or don't improve within 3 days",
        ],
    },
    "Dimorphic hemorrhoids(piles)": {
        "home_care": [
            "Increase dietary fiber intake (fruits, vegetables, whole grains)",
            "Drink plenty of water (8-10 glasses daily)",
            "Avoid straining during bowel movements",
            "Take warm baths (sitz baths) for 10-15 minutes several times a day",
            "Use moist wipes instead of dry toilet paper",
            "Avoid prolonged sitting on the toilet",
        ],
        "otc": [
            "Over-the-counter hemorrhoid creams (e.g., Preparation H)",
            "Witch hazel pads for soothing relief",
            "Stool softeners if needed (e.g., docusate sodium)",
            "Pain relievers like acetaminophen",
        ],
        "when_to_see_doctor": [
            "If bleeding is heavy or persistent",
            "If pain is severe or interfering with daily activities",
            "If home treatments don't improve symptoms within a week",
            "For evaluation of chronic or recurring hemorrhoids",
        ],
    },
    "Vertigo (Benign paroxysmal Positional Vertigo)": {
        "home_care": [
            "Avoid sudden head movements",
            "Sit or lie down immediately when vertigo starts",
            "Avoid bright lights, loud noises, and rapid movements",
            "Sleep with your head slightly elevated",
            "Perform Epley maneuver (canalith repositioning) as demonstrated by a doctor",
        ],
        "otc": [
            "Meclizine (Dramamine) for dizziness relief",
            "Ginger supplements or ginger tea for nausea",
        ],
        "when_to_see_doctor": [
            "If vertigo is severe or persistent",
            "If accompanied by hearing loss or tinnitus (ringing in ears)",
            "If you experience double vision, slurred speech, or difficulty walking",
            "For Epley maneuver demonstration by a physical therapist",
        ],
    },
    "Cervical spondylosis": {
        "home_care": [
            "Apply ice packs for the first 48 hours, then warm compresses",
            "Use a supportive pillow for sleeping",
            "Maintain good posture while sitting and standing",
            "Take frequent breaks if working at a desk",
            "Gentle neck stretches and exercises (avoid sudden movements)",
        ],
        "otc": [
            "Acetaminophen or ibuprofen for pain relief",
            "Topical pain relief creams (e.g., diclofenac gel)",
            "Neck brace/collar for short-term support (consult doctor first)",
        ],
        "when_to_see_doctor": [
            "If pain is severe or lasting more than a week",
            "If you have numbness, tingling, or weakness in arms or hands",
            "If you have difficulty walking or loss of bladder/bowel control",
            "For physical therapy evaluation",
        ],
    },
    "Tuberculosis": {
        "home_care": [
            "Take all prescribed medications exactly as directed — TB requires months of treatment",
            "Cover your mouth and nose when coughing or sneezing",
            "Wear a mask in public until no longer contagious (as advised by doctor)",
            "Get adequate rest and nutrition",
            "Avoid close contact with others until treatment is established",
        ],
        "otc": [
            "Acetaminophen for fever and body aches",
            "Cough drops or lozenges for symptom relief",
        ],
        "when_to_see_doctor": [
            "TB requires immediate medical treatment — see a doctor urgently",
            "If you have persistent cough (3+ weeks), fever, night sweats, or weight loss",
            "If you've been exposed to someone with active TB",
            "For completion of the full treatment course (6-9 months)",
        ],
    },
    "Hyperthyroidism": {
        "home_care": [
            "Eat a well-balanced diet with adequate calories",
            "Avoid caffeine and stimulants",
            "Practice stress reduction techniques (yoga, meditation)",
            "Get adequate rest and sleep",
            "Avoid vigorous exercise until symptoms are controlled",
        ],
        "otc": [
            "Beta-blockers may be prescribed — take as directed",
            "Avoid supplements containing iodine",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of hyperthyroidism (weight loss, rapid heartbeat, sweating, anxiety)",
            "For thyroid function testing (TSH, T3, T4)",
            "For treatment including antithyroid medications or other therapies",
        ],
    },
    "Malaria": {
        "home_care": [
            "Get immediate medical attention — malaria can be serious",
            "Drink plenty of fluids to stay hydrated",
            "Get adequate rest",
            "Use mosquito nets to prevent further mosquito bites",
        ],
        "otc": [
            "Acetaminophen for fever reduction",
            "Antimalarial medications must be prescribed by a doctor",
        ],
        "when_to_see_doctor": [
            "Seek medical attention immediately if you suspect malaria",
            "Especially if you have traveled to a malaria-endemic area",
            "If you have fever, chills, and flu-like symptoms after travel",
        ],
    },
    "Gastroenteritis": {
        "home_care": [
            "Drink clear fluids frequently (water, clear broths, oral rehydration solutions)",
            "Avoid solid foods for the first few hours",
            "Eat bland, easy-to-digest foods when ready (BRAT diet: Bananas, Rice, Applesauce, Toast)",
            "Avoid dairy, caffeine, alcohol, and fatty/spicy foods",
            "Wash hands thoroughly and frequently",
        ],
        "otc": [
            "Oral rehydration salts (ORS) — available at pharmacies",
            "Loperamide (Imodium) for diarrhea — but not if you have fever or bloody stool",
            "Acetaminophen for fever or body aches",
        ],
        "when_to_see_doctor": [
            "If symptoms last more than 3 days",
            "If you have severe abdominal pain",
            "If you have bloody stools or high fever (above 101°F / 38.5°C)",
            "If you show signs of dehydration (dry mouth, reduced urination, dizziness)",
        ],
    },
    "Osteoarthritis": {
        "home_care": [
            "Maintain a healthy weight to reduce joint stress",
            "Exercise regularly with low-impact activities (swimming, walking, cycling)",
            "Apply heat or cold packs to affected joints",
            "Use supportive shoes with good cushioning",
            "Consider using a cane or walker if needed",
        ],
        "otc": [
            "Acetaminophen for pain relief",
            "Topical analgesics (capsaicin cream, diclofenac gel)",
            "Glucosamine and chondroitin supplements (evidence mixed — may help some)",
        ],
        "when_to_see_doctor": [
            "If pain is severe or affecting daily activities",
            "If joints are swollen, red, or warm to touch",
            "For physical therapy evaluation",
            "If over-the-counter treatments are not effective",
        ],
    },
    "Heart attack": {
        "home_care": [
            "Call emergency services (911/108) IMMEDIATELY — do not wait",
            "Stop all activity and sit or lie down",
            "If prescribed, take one aspirin (chew and swallow)",
            "If you have nitroglycerin, take as directed",
            "Loosen tight clothing",
        ],
        "otc": [
            "Aspirin (325mg) — chew and swallow if suspected heart attack and not allergic",
        ],
        "when_to_see_doctor": [
            "Call emergency services IMMEDIATELY — this is life-threatening",
            "Do not drive yourself to the hospital",
        ],
    },
    "Dengue": {
        "home_care": [
            "Get plenty of rest",
            "Drink lots of water and fluids to prevent dehydration",
            "Use mosquito nets to prevent further mosquito bites",
            "Monitor temperature regularly",
        ],
        "otc": [
            "Acetaminophen for fever and pain relief",
            "AVOID ibuprofen, aspirin, and NSAIDs — they increase bleeding risk",
        ],
        "when_to_see_doctor": [
            "Seek medical attention for proper diagnosis and monitoring",
            "If fever is very high (above 104°F / 40°C)",
            "If you experience severe abdominal pain, persistent vomiting, or bleeding gums",
            "Warning signs: severe abdominal pain, vomiting, difficulty breathing, cold extremities",
        ],
    },
    "Pneumonia": {
        "home_care": [
            "Get plenty of rest",
            "Drink lots of fluids",
            "Use a humidifier to ease breathing",
            "Do not smoke or vape",
            "Take deep breaths and cough to clear lungs (as tolerated)",
        ],
        "otc": [
            "Acetaminophen or ibuprofen for fever and pain",
            "Cough suppressants — but only if recommended by doctor",
        ],
        "when_to_see_doctor": [
            "Pneumonia requires medical attention — see a doctor",
            "If you have difficulty breathing, chest pain, or high fever",
            "If you are over 65, have chronic conditions, or a weakened immune system",
            "If symptoms don't improve after 3 days",
        ],
    },
    "Urinary tract infection": {
        "home_care": [
            "Drink plenty of water (aim for 8+ glasses daily)",
            "Urinate frequently and don't hold it in",
            "Use a heating pad on your lower abdomen for pain relief",
            "Wipe front to back after using the bathroom",
            "Avoid caffeine, alcohol, and spicy foods",
        ],
        "otc": [
            "Phenazopyridine (Azo) for urinary pain relief",
            "Acetaminophen for pain/fever",
            "Cranberry supplements (may help prevent but not treat active infection)",
        ],
        "when_to_see_doctor": [
            "If you have burning pain during urination or frequent urge to urinate",
            "If you have fever, back pain, or blood in urine",
            "UTIs require antibiotic treatment — see a doctor",
            "If symptoms persist after treatment",
        ],
    },
    "Hypoglycemia": {
        "home_care": [
            "Eat or drink 15g of fast-acting carbohydrates immediately (glucose tablets, fruit juice, regular soda, or candy)",
            "Wait 15 minutes and recheck blood sugar",
            "If still low, repeat treatment",
            "Once stabilized, eat a balanced meal or snack with protein and complex carbs",
            "If diabetic, always carry glucose tablets or sugary snacks",
        ],
        "otc": [
            "Glucose tablets or gel (available at pharmacies)",
            "Glucagon emergency kit (if prescribed by doctor)",
        ],
        "when_to_see_doctor": [
            "If you have recurrent hypoglycemic episodes",
            "If you lose consciousness (requires emergency glucagon injection)",
            "If you are not diabetic but experience low blood sugar symptoms",
            "For adjustment of diabetes medications if applicable",
        ],
    },
    "Bronchial Asthma": {
        "home_care": [
            "Avoid known triggers (allergens, smoke, cold air, exercise-induced triggers)",
            "Use a peak flow meter to monitor lung function",
            "Keep rescue inhaler (e.g., albuterol) accessible at all times",
            "Follow your asthma action plan as provided by your doctor",
            "Keep your home clean and free of dust mites and mold",
        ],
        "otc": [
            "Rescue inhalers require a prescription",
            "Allergy medications may help with allergen-triggered asthma",
            "Consult doctor before taking any new medications",
        ],
        "when_to_see_doctor": [
            "If you need rescue inhaler more than 2 times per week",
            "If you have difficulty breathing that doesn't improve with medication",
            "If you have frequent nighttime awakenings due to asthma symptoms",
            "For regular asthma check-ups and medication adjustment",
        ],
    },
    "Arthritis": {
        "home_care": [
            "Exercise regularly with low-impact activities (swimming, walking, tai chi)",
            "Apply heat to stiff joints and cold to inflamed joints",
            "Maintain a healthy weight",
            "Use assistive devices if needed (jar openers, long-handled tools)",
            "Eat an anti-inflammatory diet (fruits, vegetables, omega-3 fatty acids)",
        ],
        "otc": [
            "Acetaminophen for pain relief",
            "Ibuprofen or naproxen for pain and inflammation (consult pharmacist for appropriate use)",
            "Topical pain relief creams (capsaicin, diclofenac gel)",
        ],
        "when_to_see_doctor": [
            "If joint pain is persistent or worsening",
            "If joints are swollen, red, hot, or stiff in the morning",
            "For diagnosis and treatment plan",
            "If over-the-counter treatments aren't helping",
        ],
    },
    "Hepatitis D": {
        "home_care": [
            "Get plenty of rest",
            "Eat a healthy diet with adequate nutrition",
            "Avoid alcohol completely",
            "Stay hydrated",
        ],
        "otc": [
            "Consult doctor before taking any medications",
            "Avoid acetaminophen at high doses",
        ],
        "when_to_see_doctor": [
            "Hepatitis D requires medical management — see a specialist",
            "If you have hepatitis B (HDV only occurs with HBV)",
            "For liver function monitoring and antiviral treatment options",
        ],
    },
    "Hypothyroidism": {
        "home_care": [
            "Take thyroid medication at the same time every day (usually in morning on empty stomach)",
            "Do not skip medication doses",
            "Eat a balanced diet with adequate iodine (but don't overdo iodine supplements)",
            "Exercise regularly to boost metabolism",
            "Get adequate sleep",
        ],
        "otc": [
            "Thyroid hormone replacement requires a prescription",
            "Avoid taking calcium or iron supplements within 4 hours of thyroid medication",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of hypothyroidism (fatigue, weight gain, cold intolerance, dry skin)",
            "For thyroid function testing (TSH levels)",
            "For regular monitoring and medication adjustment",
        ],
    },
    "Acne": {
        "home_care": [
            "Wash face twice daily with a gentle cleanser",
            "Avoid touching or picking at pimples",
            "Use non-comedogenic (non-pore-clogging) products",
            "Avoid harsh scrubbing",
            "Change pillowcases regularly",
            "Keep hair clean and away from face",
        ],
        "otc": [
            "Benzoyl peroxide creams/gels",
            "Salicylic acid cleansers or pads",
            "Topical retinoid creams (adapalene/Differin now available OTC)",
        ],
        "when_to_see_doctor": [
            "If over-the-counter treatments aren't working after 8-12 weeks",
            "If acne is severe, painful, or causing scarring",
            "For prescription treatments (antibiotics, stronger retinoids, or hormonal therapy)",
        ],
    },
    "GERD": {
        "home_care": [
            "Eat smaller, more frequent meals",
            "Avoid lying down for 2-3 hours after eating",
            "Elevate the head of your bed by 6-8 inches",
            "Avoid trigger foods (spicy, fatty, acidic foods, chocolate, caffeine, alcohol)",
            "Maintain a healthy weight",
            "Don't smoke or vape",
        ],
        "otc": [
            "Antacids (Tums, Rolaids) for immediate relief",
            "H2 blockers (famotidine/Pepcid, ranitidine)",
            "Proton pump inhibitors (omeprazole/Prilosec, lansoprazole)",
        ],
        "when_to_see_doctor": [
            "If symptoms persist despite OTC treatment for 2 weeks",
            "If you have difficulty swallowing or pain with swallowing",
            "If you have unintended weight loss",
            "If you have frequent heartburn (2+ times per week)",
        ],
    },
    "Peptic ulcer disease": {
        "home_care": [
            "Avoid NSAIDs (ibuprofen, aspirin, naproxen) as they can worsen ulcers",
            "Avoid alcohol and smoking",
            "Eat small, frequent meals",
            "Avoid spicy and acidic foods",
            "Manage stress levels",
        ],
        "otc": [
            "Antacids for symptom relief",
            "H2 blockers or PPIs for acid reduction",
        ],
        "when_to_see_doctor": [
            "If you have persistent abdominal pain, especially between meals or at night",
            "If you have black/tarry stools or vomit blood",
            "If you experience sudden, severe abdominal pain",
            "For H. pylori testing and treatment",
        ],
    },
    "Psoriasis": {
        "home_care": [
            "Keep skin moisturized with thick creams or ointments",
            "Take short, warm (not hot) baths with colloidal oatmeal or Epsom salts",
            "Avoid triggers like stress, infections, and skin injuries",
            "Get moderate sun exposure (use sunscreen on unaffected areas)",
            "Avoid alcohol and smoking",
        ],
        "otc": [
            "Moisturizers with ceramides or hyaluronic acid",
            "Topical hydrocortisone for mild cases",
            "Coal tar shampoos or creams",
            "Salicylic acid for scale removal",
        ],
        "when_to_see_doctor": [
            "If OTC treatments aren't effective",
            "If psoriasis is covering large areas of body",
            "If you have joint pain (possible psoriatic arthritis)",
            "For prescription topical treatments, phototherapy, or systemic medications",
        ],
    },
    "Drug Reaction": {
        "home_care": [
            "Stop taking the suspected medication immediately (unless it's essential — consult doctor first)",
            "Apply cool compresses to itchy skin",
            "Avoid scratching",
            "Keep a record of which medication caused the reaction",
        ],
        "otc": [
            "Antihistamines (diphenhydramine/Benadryl, cetirizine/Zyrtec) for itching",
            "Topical hydrocortisone cream for localized reactions",
        ],
        "when_to_see_doctor": [
            "If you have difficulty breathing, swelling of face/lips/tongue — call emergency services",
            "If rash is widespread or severe",
            "If you have fever, joint pain, or blistering skin",
            "For alternative medication recommendations",
        ],
    },
    "Diabetes": {
        "home_care": [
            "Monitor blood sugar regularly as advised by your doctor",
            "Follow a balanced diet low in refined sugars and carbohydrates",
            "Exercise regularly (30 minutes daily, most days)",
            "Maintain a healthy weight",
            "Check feet daily for cuts, blisters, or sores",
            "Stay hydrated with water",
        ],
        "otc": [
            "Blood glucose monitor and test strips",
            "Consult doctor before taking any supplements",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of diabetes (frequent urination, excessive thirst, unexplained weight loss)",
            "For diabetes screening if you have risk factors (family history, overweight, age 45+)",
            "For regular A1C monitoring and medication management",
            "If blood sugar is consistently high or very low",
        ],
    },
    "Varicose veins": {
        "home_care": [
            "Elevate your legs when sitting or lying down",
            "Avoid standing or sitting for long periods",
            "Exercise regularly (walking, swimming, cycling)",
            "Wear compression stockings as recommended",
            "Maintain a healthy weight",
            "Avoid crossing your legs when sitting",
        ],
        "otc": [
            "Compression stockings (graduated compression) — available at pharmacies",
            "Horse chestnut extract (some evidence for symptom relief)",
        ],
        "when_to_see_doctor": [
            "If veins are painful, swollen, or causing skin changes",
            "If you develop leg ulcers or skin discoloration near ankles",
            "If you have sudden leg swelling or pain (possible blood clot)",
            "For evaluation of treatment options (sclerotherapy, laser, surgery)",
        ],
    },
    "Hepatitis A": {
        "home_care": [
            "Get plenty of rest",
            "Stay hydrated with water and clear fluids",
            "Avoid alcohol completely",
            "Eat small, frequent meals",
            "Avoid fatty foods",
            "Wash hands thoroughly after bathroom use (highly contagious)",
        ],
        "otc": [
            "Acetaminophen for fever — use cautiously and at low doses",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of hepatitis (fatigue, nausea, abdominal pain, jaundice)",
            "If you've been exposed to hepatitis A (vaccination can prevent if given within 2 weeks)",
            "For hepatitis A vaccine if traveling to endemic areas",
        ],
    },
    "Hepatitis E": {
        "home_care": [
            "Get plenty of rest",
            "Stay hydrated with water and clear fluids",
            "Avoid alcohol completely",
            "Eat a healthy diet",
        ],
        "otc": [
            "Acetaminophen for fever — use cautiously",
        ],
        "when_to_see_doctor": [
            "If you have symptoms of hepatitis",
            "Especially important for pregnant women (can be severe)",
            "For proper diagnosis and monitoring",
        ],
    },
    "Migraine": {
        "home_care": [
            "Rest in a dark, quiet room",
            "Apply a cold pack to your forehead",
            "Drink water if dehydrated",
            "Identify and avoid trigger factors (certain foods, stress, lack of sleep)",
            "Maintain a regular sleep schedule",
            "Practice relaxation techniques",
        ],
        "otc": [
            "Acetaminophen, ibuprofen, or naproxen for mild to moderate migraines",
            "Caffeine (in small amounts) may help — found in some migraine combination products",
            "Motion sickness bands for nausea relief",
        ],
        "when_to_see_doctor": [
            "If migraines are frequent (4+ per month) or severe",
            "If OTC medications aren't effective",
            "If you have a sudden, severe headache unlike your usual migraines",
            "For prescription migraine medications or preventive treatment",
        ],
    },
    "Allergy": {
        "home_care": [
            "Identify and avoid allergens (pollen, dust, pet dander, certain foods)",
            "Keep windows closed during high pollen seasons",
            "Use air purifiers with HEPA filters",
            "Wash bedding in hot water weekly",
            "Shower and change clothes after being outdoors",
        ],
        "otc": [
            "Antihistamines (cetirizine/Zyrtec, loratadine/Claritin, fexofenadine/Allegra)",
            "Nasal corticosteroid sprays (fluticasone/Flonase)",
            "Decongestants (pseudoephedrine/Sudafed) — short-term use only",
            "Eye drops for allergic conjunctivitis",
        ],
        "when_to_see_doctor": [
            "If OTC medications aren't controlling symptoms",
            "If allergies are interfering with daily life or sleep",
            "For allergy testing to identify specific triggers",
            "For allergy shots (immunotherapy) if appropriate",
        ],
    },
    "Jaundice": {
        "home_care": [
            "Stay hydrated with water and clear fluids",
            "Get plenty of rest",
            "Avoid alcohol completely",
            "Eat a healthy, balanced diet",
        ],
        "otc": [
            "Consult doctor before taking any medications",
        ],
        "when_to_see_doctor": [
            "Jaundice requires medical evaluation — see a doctor",
            "If you have yellowing of skin or eyes",
            "If you have abdominal pain, fever, or dark urine",
            "For liver function testing and diagnosis",
        ],
    },
    "AIDS": {
        "home_care": [
            "Take antiretroviral therapy (ART) exactly as prescribed — consistently and without skipping",
            "Eat a nutritious diet to maintain immune function",
            "Practice safe sex to prevent transmission",
            "Get regular medical check-ups for CD4 count and viral load monitoring",
            "Avoid infections by washing hands frequently and avoiding sick contacts",
        ],
        "otc": [
            "Consult doctor before taking any medications or supplements",
        ],
        "when_to_see_doctor": [
            "See a doctor for HIV testing if you think you may have been exposed",
            "For starting and maintaining antiretroviral therapy",
            "For regular monitoring and preventive care",
            "Immediately if you develop fever, difficulty breathing, or severe symptoms",
        ],
    },
    "Alcoholic hepatitis": {
        "home_care": [
            "Stop drinking alcohol completely — this is the most important step",
            "Get plenty of rest",
            "Eat a nutritious diet",
            "Stay hydrated",
            "Join a support group for alcohol cessation if needed",
        ],
        "otc": [
            "Consult doctor before taking any medications (especially acetaminophen)",
            "Vitamin B and folic acid supplements may be recommended",
        ],
        "when_to_see_doctor": [
            "Alcoholic hepatitis requires medical evaluation — see a doctor",
            "If you have jaundice, abdominal swelling, or confusion",
            "For liver function testing and treatment",
            "For support with alcohol cessation programs",
        ],
    },
}

# ===== Severity Mapping =====
EMERGENCY_DISEASES = ["Heart attack", "Paralysis (brain hemorrhage)"]
SEVERE_DISEASES = ["Pneumonia", "Tuberculosis", "Malaria", "Dengue", "Typhoid", "Alcoholic hepatitis", "AIDS"]
MODERATE_DISEASES = ["Hepatitis B", "Hepatitis C", "Hepatitis A", "Hepatitis D", "Hepatitis E", "Chronic cholestasis", "Hypertension", "Diabetes", "Hyperthyroidism", "Hypothyroidism"]


def check_emergency_symptoms(text: str) -> bool:
    """Check if the input contains emergency red-flag symptoms."""
    if not text:
        return False
    text_lower = text.lower()
    for symptom in EMERGENCY_SYMPTOMS:
        if symptom in text_lower:
            return True
    return False


def get_severity(disease_name: str) -> str:
    """Determine severity level for a disease."""
    if disease_name in EMERGENCY_DISEASES:
        return "emergency"
    if disease_name in SEVERE_DISEASES:
        return "severe"
    if disease_name in MODERATE_DISEASES:
        return "moderate"
    return "mild"


def get_recommendations(disease_name: str) -> dict:
    """Get recommendations for a disease."""
    return DISEASE_RECOMMENDATIONS.get(disease_name, {
        "home_care": ["Rest and stay hydrated", "Monitor your symptoms", "Get adequate sleep"],
        "otc": ["Consult a pharmacist for appropriate OTC medications", "Do not self-prescribe antibiotics"],
        "when_to_see_doctor": ["If symptoms persist for more than 3-5 days", "If symptoms worsen or become severe", "If you develop new concerning symptoms"],
    })


def load_model_bundle() -> Dict[str, object]:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            bundle = pickle.load(handle)
            models = bundle.get("models", {})
            supports_probabilities = all(
                hasattr(model, "predict_proba") for model in models.values()
            )
            if supports_probabilities:
                return bundle

    return train_and_save_model(MODEL_PATH)


MODEL_CACHE = load_model_bundle()


def normalize_symptoms(text: str) -> List[str]:
    if not text:
        return []

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return []

    symptoms = set()
    for phrase, canonical_name in SYMPTOM_ALIASES.items():
        if phrase in normalized:
            symptoms.add(canonical_name)

    for feature in MODEL_CACHE["features"]:
        feature_text = feature.replace("_", " ")
        if feature_text in normalized:
            symptoms.add(feature)

    return sorted(symptoms)


def build_feature_vector(symptoms: List[str]) -> List[int]:
    feature_index = {feature: idx for idx, feature in enumerate(MODEL_CACHE["features"])}
    feature_vector = [0] * len(MODEL_CACHE["features"])
    for symptom in symptoms:
        if symptom in feature_index:
            feature_vector[feature_index[symptom]] = 1
    return feature_vector


def build_top_predictions(feature_frame: pd.DataFrame) -> List[Dict[str, object]]:
    models = MODEL_CACHE["models"]
    model_names = ["rf", "nb", "svm"]
    class_ids = sorted({class_id for model_name in model_names for class_id in models[model_name].classes_.tolist()})

    averaged_probabilities = []
    for class_id in class_ids:
        total_probability = 0.0
        contributing_models = 0
        for model_name in model_names:
            model = models[model_name]
            if class_id in model.classes_:
                class_index = list(model.classes_).index(class_id)
                total_probability += float(model.predict_proba(feature_frame)[0][class_index])
                contributing_models += 1

        if contributing_models:
            averaged_probabilities.append((class_id, total_probability / contributing_models))

    averaged_probabilities.sort(key=lambda item: item[1], reverse=True)

    top_predictions = []
    for class_id, probability in averaged_probabilities[:3]:
        disease_name = MODEL_CACHE["encoder"].inverse_transform([class_id])[0]
        top_predictions.append({
            "disease": disease_name,
            "confidence": round(probability * 100, 2),
        })

    return top_predictions


def predict_from_text(text: str) -> Dict[str, object]:
    symptoms = normalize_symptoms(text)
    if not symptoms:
        return {
            "response": "Please share one or more recognizable symptoms such as fever, headache, or skin rash.",
            "predicted_disease": None,
            "symptoms": [],
            "accuracy": MODEL_CACHE["accuracy"],
            "top_predictions": [],
            "note": "These confidence scores represent the model's prediction confidence based on the input symptoms. They are not a substitute for a medical diagnosis.",
        }

    feature_vector = build_feature_vector(symptoms)
    df = pd.DataFrame([feature_vector], columns=MODEL_CACHE["features"])

    models = MODEL_CACHE["models"]
    rf_pred = int(models["rf"].predict(df)[0])
    nb_pred = int(models["nb"].predict(df)[0])
    svm_pred = int(models["svm"].predict(df)[0])

    final_pred = int(pd.Series([rf_pred, nb_pred, svm_pred]).mode().iloc[0])
    predicted_disease = MODEL_CACHE["encoder"].inverse_transform([final_pred])[0]
    top_predictions = build_top_predictions(df)
    if top_predictions and top_predictions[0]["disease"] != predicted_disease:
        top_predictions = [{"disease": predicted_disease, "confidence": top_predictions[0]["confidence"]}] + [entry for entry in top_predictions if entry["disease"] != predicted_disease][:2]

    response = (
        f"I think the most likely disease is {predicted_disease}. "
        f"This is based on the symptoms you mentioned: {', '.join(symptoms)}."
    )

    # Store in session history (only if there's an active request context)
    try:
        if "history" not in session:
            session["history"] = []
        history_entry = {
            "symptoms_input": text,
            "symptoms": symptoms,
            "predicted_disease": predicted_disease,
            "confidence": top_predictions[0]["confidence"] if top_predictions else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        session["history"] = session["history"] + [history_entry]
        session.modified = True
    except RuntimeError:
        pass  # No request context (e.g., during testing)

    return {
        "response": response,
        "predicted_disease": predicted_disease,
        "symptoms": symptoms,
        "accuracy": MODEL_CACHE["accuracy"],
        "top_predictions": top_predictions,
        "note": "These confidence scores represent the model's prediction confidence based on the input symptoms. They are not a substitute for a medical diagnosis.",
    }


# ===== Routes =====

@app.route("/")
def index():
    return redirect(url_for("welcome"))


@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/symptoms")
def symptoms_page():
    return render_template("symptoms.html")


@app.route("/results")
def results_page():
    disease = request.args.get("disease", "")
    symptoms_text = request.args.get("symptoms", "")
    predictions = []
    symptoms_list = []
    severity = "mild"

    if symptoms_text:
        result = predict_from_text(symptoms_text)
        predictions = result.get("top_predictions", [])
        symptoms_list = result.get("symptoms", [])
        if predictions:
            severity = get_severity(predictions[0]["disease"])
    elif disease:
        recs = get_recommendations(disease)
        predictions = [{"disease": disease, "confidence": 0}]
        severity = get_severity(disease)

    return render_template(
        "results.html",
        predictions=predictions,
        symptoms=symptoms_list,
        severity=severity,
    )


@app.route("/recommendations")
def recommendations_page():
    disease = request.args.get("disease", "")
    confidence = request.args.get("confidence", "0")
    recs = {}
    if disease:
        recs = get_recommendations(disease)
    return render_template(
        "recommendations.html",
        disease=disease,
        confidence=confidence,
        recommendations=recs,
    )


@app.route("/history")
def history_page():
    history = session.get("history", [])
    return render_template("history.html", history=history)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["history"] = []
    session.modified = True
    return jsonify({"status": "ok"})


@app.route("/emergency")
def emergency_page():
    return render_template("emergency.html")


@app.route("/about")
def about_page():
    features = MODEL_CACHE.get("features", [])
    encoder = MODEL_CACHE.get("encoder", None)
    num_diseases = len(encoder.classes_) if encoder else 0
    accuracy = MODEL_CACHE.get("accuracy", {})
    return render_template(
        "about.html",
        num_diseases=num_diseases,
        num_symptoms=len(features),
        symptoms_list=features,
        accuracy=accuracy,
    )


@app.route("/profile", methods=["GET", "POST"])
def profile_page():
    saved = False
    if request.method == "POST":
        session["profile"] = {
            "age": request.form.get("age", ""),
            "gender": request.form.get("gender", ""),
            "existing_conditions": request.form.get("existing_conditions", ""),
            "allergies": request.form.get("allergies", ""),
        }
        session.modified = True
        saved = True
    profile = session.get("profile", {})
    return render_template("profile.html", profile=profile, saved=saved)


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")

    # Check for emergency symptoms
    is_emergency = check_emergency_symptoms(message)
    result = predict_from_text(message)

    if is_emergency:
        result["emergency"] = True
        result[
            "response"
        ] = "🚨 The symptoms you described may indicate a medical emergency. Please call emergency services (911/108) immediately."

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
