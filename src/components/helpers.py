import os 
import sys

sys.path.append(os.path.abspath(os.curdir))

import torch as t
import json
from typing import List
from transformers import PreTrainedTokenizer
from src.utils.settings import Settings
from transformers import AutoTokenizer 
from fastchat.conversation import get_conv_template


SETTINGS = Settings()

# Directories 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTORS_PATH = os.path.join(BASE_DIR, "vectors", "without_norm_vectors")
NORMALIZED_VECTORS_PATH = os.path.join(BASE_DIR, "vectors", "normalized_vectors")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data")
ACTIVATIONS_PATH = os.path.join(BASE_DIR, "results","activations")


# Jailbreak selections
ALL_WRAPPERS = ['prefix_injection', 'few_shot_json', 'disemvowel', 'leetspeak', 'wikipedia_with_title', 
                'wikipedia', 'payload_split', 'refusal_suppression', 'dev_mode_v2', 'refusal_suppression_inv', 
                'italian', 'adverserial_suffix', 'style_injection_short', 'poems', 'prefix_injection_hello', 
                'eng_question_it_output', 'distractors_negated', 'auto_obfuscation', 'evil_confidant', 'distractors', 
                'AIM', 'eng_it_merged', 'it_question_eng_output', 'style_injection_json']

POTENT_WRAPPERS = ['refusal_suppression', 'dev_mode_v2','style_injection_short', 
                   'evil_confidant', 'distractors', 'AIM',]

MIDDLE_WRAPPERS = [ 'wikipedia_with_title', 'prefix_injection','style_injection_json','poems', 
                   'payload_split','few_shot_json']

    
# Get paths
def get_results_dir(behavior: str) -> str:
    return os.path.join(RESULTS_PATH, behavior)

def get_activations_dir() -> str:
    return ACTIVATIONS_PATH

def get_activations_path(layer, model_abb:str, wrapper, variable_element=None
) -> str:
    if variable_element:
        path = os.path.join(
        get_activations_dir(),
        f"activations_{wrapper}_{make_tensor_save_suffix(layer, model_abb)}", f"_{variable_element}.pt"
    )
        
    path = os.path.join(
        get_activations_dir(),
        f"activations_{wrapper}_{make_tensor_save_suffix(layer, model_abb)}.pt", 
    )
    return path

def get_activations(layer, model_abb, wrapper, variable_element=None):  
    return t.load(get_activations_path(layer, model_abb, wrapper, variable_element))  

def get_vector_dir(normalized=False) -> str:
    return os.path.join(NORMALIZED_VECTORS_PATH if normalized else VECTORS_PATH)

def get_vector_path(layer, model_name_path: str, wrapper,  normalized=False, variable_element = None) -> str:
    if variable_element: 
        path = os.path.join(
            get_vector_dir(normalized=normalized),
        f"vec_{wrapper}_layer_{make_tensor_save_suffix(layer, model_name_path)}_{variable_element}.pt")
        #print("\n\nI am in the variable element part\n\n")
    
    else:    
        path = os.path.join(
            get_vector_dir(normalized=normalized),
            f"vec_{wrapper}_layer_{make_tensor_save_suffix(layer, model_name_path)}.pt",)
    return path

def get_vector(layer, model_name_path, wrapper, normalized=False, variable_element=None):
    return t.load(get_vector_path(layer, model_name_path, wrapper, normalized, variable_element))

def make_tensor_save_suffix(layer, model_abb):
    return f'{layer}_{model_abb.split("/")[-1]}'

  
# Other helper functions
def find_last_subtensor_position(tensor, sub_tensor):
    n,m = tensor.size(0), sub_tensor.size(0)
    
    if m>n: 
        return -1
    
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i:i+m], sub_tensor):
            return i
    return -1
    
def find_instruction_end_position(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) -1

def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None: 
        after_id = position_ids.min().item() -1 
    mask = position_ids >= after_id 
    mask = mask.unsqueeze(-1) 
    matrix += mask.float() * vector 
    return matrix


B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS_Question, EOS_Question, BOS_Answer, EOS_Answer = SETTINGS.data["general"]['prompt_bos_eos']
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"
ADD_AFTER_POS_CHAT = "[\INST]"
ADD_AFTER_POS_VICUNA = "\n### Assistant:"

def tokenize_vicuna(tokenizer: PreTrainedTokenizer,
                    user_input: str, 
                    model_output: str = None,
                    with_sys_prompt:bool = True):
        
    if with_sys_prompt == False: 
        input_content = f"USER: {user_input.strip()} \n ASSISTANT:"
        return input_content
     
    conv = get_conv_template("zero_shot")
    conv.append_message(conv.roles[0], user_input)
 
    if model_output is not None:
        conv.append_message(conv.roles[1], model_output.strip())
    else:
        conv.append_message(conv.roles[1], None)
        
    input_content = conv.get_prompt()
    return tokenizer.encode(input_content)

def tokenize_llama_chat(tokenizer: PreTrainedTokenizer, 
                        user_input: str,
                        model_output: str = None,
                        system_prompt:str = None,) -> List[int]: 
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    
    input_content += f"{BOS_Question} {user_input.strip()} {EOS_Question}" 
    
    if model_output is not None:
        input_content += f"\n\n {BOS_Answer} {model_output.strip()} {EOS_Answer}" 
    print(f"I feed into the model the following input: {input_content}")
    return tokenizer.encode(input_content)

def get_json_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def get_figures_dir():
    return "./results/figures"

def extract_unique_wrappers(data_path="./data/vector_dataset.json"):
    with open(data_path, "r") as f:
        data = json.load(f)
    unique_wrappers = set(item["wrapper"] for item in data)

    # Convert set to list if needed
    unique_wrappers_list = list(unique_wrappers)
    return unique_wrappers_list

def make_equal_len_sequences(completion, refusal, model_name):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the completion and refusal sequences
    completion_tokens = tokenizer.tokenize(completion)
    refusal_tokens = tokenizer.tokenize(refusal)

    # Find the length of each tokenized sequence
    completion_length = len(completion_tokens)
    refusal_length = len(refusal_tokens)

    # Find the size of the shortest sequence
    shortest_length = min(completion_length, refusal_length)

    # Shorten the longer sequence to match the length of the shortest one
    completion_shortened_tokens = completion_tokens[:shortest_length]
    refusal_shortened_tokens = refusal_tokens[:shortest_length]

    # Convert tokenized sequences back to normal language
    completion_shortened = tokenizer.convert_tokens_to_string(completion_shortened_tokens)
    refusal_shortened = tokenizer.convert_tokens_to_string(refusal_shortened_tokens)

    return completion_shortened, refusal_shortened


UNIQUE_PROMPTS_TRAIN = ['self_harm', 'swatting_guide', 'racial_supremacy', 'obtain_stolen_goods',
                        'exploiting_vulnerabilities', 'intentional_car_crash',
                        'best_identity_theft', 'human_trafficking_network', 'stealing',
                        'plagiarism_techniques', 'illegal_immigration_guide', 'swatting',
                        'violent_protest_tactics', 'violate_privacy', 'murder_plot',
                        'undermining_relationship', 'terrorist_targets', 'design_virus',
                        'faking_kidnapping', 'destructive_hacking', 'malicious_hacking_tool',
                        'workplace_sabotage', 'sexual_exploitation', 'meth_synthesis',
                        'building_illegal_radio_transmitter', 'spread_rumors',
                        'defrauding_insurance', 'illegal_drug_services', 'ddos_attack',
                        'scamming_strategies', 'online_grooming', 'vote_manipulation',
                        'burning_evidence', 'drug_dealing_tips', 'offensive_limericks',
                        'hate_speech_2', 'hacking_tutorials', 'nuclear_weapon_design',
                        'hacking_into_security_system', 'fake_id_documents', 'poison_diversity',
                        'organised_crime_involvement', 'election_misinformation', 'maximizing_pain',
                        'untraceable_poisons', 'create_drug', 'torture_techniques',
                        'extremist_propaganda', 'ransomware', 'human_kidnapping',
                        'terrorist_attack_ideas', 'dating_profile', 'hitman', 'cannabis_manufacture',
                        'school_shooting_plan', 'nuclear_bomb_blueprint', 'toxic_substances',
                        'drunk_driving_avoid_arrest', 'darkweb_markets', 'ethnic_stereotypes',
                        'offensive_meme_creation', 'blame_someone', 'kidnapping_strategy',
                        'serving_contaminated_food', 'plausible_deniability',
                        'cyberbully_techniques', 'uranium_purchase', 'misogynistic_viewpoints',
                        'assassination', 'spread_misinformation', 'enable_stalker',
                        'illegal_migration', 'violent_game', 'avoid_child_support',
                        'terrorism_tactics', 'arson_tactics', 'fake_news', 'online_stalking_tips',
                        'violence_political_opponent', 'ransom_money', 'taking_illegal_photos',
                        'creating_a_pyramid_scheme', 'embezzlement', 'illegal_hacking',
                        'fake_nude_photos', 'terrorism_act', 'expression_banning',
                        'avoiding_cybercrimes_detection', 'counterfeit_currency',
                        'political_conspiracy', 'freedom_of_speech', 'selling_body_organs',
                        'spreading_fake_news', 'animal_abuse', 'bomb_instructions',
                        'identity_theft_guide', 'privacy_invasion', 'terrorism', 'pirated_software',
                        'cracking_security_system', 'fake_passport', 'forcing_beliefs',
                        'false_accusations', 'malicious_code_evolution', 'president_assassination',
                        'doxxing_guide', 'bio_weapon_creation', 'abusing_mom', 'child_pornography',
                        'fake_news_spreading', 'public_nudity', 'offensive_jokes', 'catfishing_tips',
                        'bypassing_filters', 'offensive_sexist_remarks',
                        'offensive_fictional_character', 'illegal_business', 'online_harassment',
                        'humiliating_someone', 'suicide_techniques', 'data_leak', 'forging_art',
                        'neo_nazi_manifesto', 'bullying_tactics', 'sex_trafficking',
                        'dangerous_diet', 'evading_police', 'drugging_individuals',
                        'human_experimentation', 'fraudulent_expenses', 'alcohol_poisoning',
                        'poisonous_plant_extraction', 'camgirl_exploitation', 'pro_ana_tips',
                        'homemade_explosive', 'politician_extortion', 'making_drugs',
                        'spreading_misinformation', 'sarin_synthesis', 'building_malware',
                        'building_an_illegal_weapon', 'underage_exploitation', 'false_documents',
                        'robbery_instructions', 'vigilante_justice', 'attractiveness_stereotypes',
                        'poison_introduction', 'illegal_immigration_routes',
                        'getting_rid_of_evidence', 'deepfake_president',
                        'serial_killing_techniques', 'misogynistic_comments',
                        'escaping_police_custody', 'totalitarian_propaganda', 'forging_documents',
                        'sexually_explicit_content', 'offensive_jokes_2',
                        'best_places_to_hide_drugs', 'cryptocurrency_scam', 'toxic_relation_advice',
                        'blackmail_methods', 'propaganda_deception', 'cheat_on_exam',
                        'cult_recruitment', 'hacking_private_information', 'hate_speech',
                        'hostage_negotiation', 'suicide_cult', 'underage_relationships', 'arson',
                        'obtaining_phones', 'exploiting_disabilities', 'organized_crime',
                        'fake_death_certificate', 'radioactive_material', 'bully_detection',
                        'prostitute_prices', 'discriminate_vaccination', 'social_engineering',
                        'deceptive_scamming', 'radioactive_materials', 'poison_spouse',
                        'drug_recipe', 'fentanyl_synthesis', 'racist_remarks', 'killing_a_witness',
                        'animal_fighting', 'firing_school_principal', 'staged_car_accident',
                        'gaslighting_techniques', 'home_invasion', 'bypassing_security_systems',
                        'burglary_tips', 'avoiding_military', 'adult_websites',
                        'nude_picture_request', 'mass_poisoning', 'making_fake_currency',
                        'racist_insults', 'unsafe_activities', 'adultery_blackmail',
                        'poisoning_nature', 'racial_excitement', 'nuclear_arsenal',
                        'bill_gates_houses', 'biohazard_spread', 'defaming_competitor',
                        'secret_torture_methods', 'create_psychoactive_drug', 'rigging_elections',
                        'forging_signature', 'discrediting_rival', 'planting_evidence',
                        'manual_strangulation', 'illegal_streaming', 'malicious_slander',
                        'emulate_edangerous_pranks_dinburgh', 'secret_cameras',
                        'shoplifting_techniques', 'deepfake_techniques',
                        'bullying_husbands_mistress', 'world_domination', 'unauthorized_access',
                        'date_rape_drug', 'child_trafficking', 'bypassing_security_system',
                        'doxxing_politician', 'ruin_someones_life', 'escape_prison',
                        'gaslighting_partner', 'damage_to_property', 'deceptive_contracts',
                        'assassination_tactics', 'date_drugging', 'problematic_twitter_bio',
                        'harassing_celebrities', 'drug_dealing', 'faking_medical_conditions',
                        'virginity_sales', 'advertise_prostitution', 'synthetic_drugs',
                        'hijacking_plans', 'nonconsensual_explicit_content', 'chemical_weapon',
                        'stalk_celebrities', 'spying_on_neighbors', 'nonconsensual_photos',
                        'blackmail_strategy', 'planting_malware', 'arson_instructions',
                        'political_extremism', 'self_harm_2', 'hacking_email',
                        'illegal_online_communities', 'create_false_identity', 'sexual_harassment',
                        'bypassing_website_blocking', 'abortion_misinformation', 'forced_marriage',
                        'fake_reviews', 'destroying_environment', 'cyber_harassment',
                        'harmful_persuasion', 'intrusive_hacking', 'revenge_porn',
                        'illegal_firearms', 'shoplifting_manual', 'political_assassinations',
                        'ilegal_experimentation', 'imprisonment_strategy', 'violent_protests',
                        'mutilation', 'hotwiring_car', 'revenge_tactics', 'evasion_authorities',
                        'homophobic_comments', 'homemade_explosives', 'evading_taxes',
                        'explicit_parenting_advice', 'child_exploitation', 'breaking_privacy',
                        'offensive_nickname']  


UNIQUE_PROMPTS_TEST = [
    'chemical_attack', 'sexually_harassing_compliments', 'bypassing_school_security', 'violent_protest_plans', 
    'child_labor', 'child_labour_purchase', 'extreme_weight_loss', 'threatening_letter', 'gaming_addiction', 
    'hacking_bank', 'vandalizing_stop_sign', 'anorexic_tips', 'blackmailing_tips', 'manipulate_emotions', 
    'stalking_tips', 'drug_production', 'berkeley_faculty_contact_info', 'drug_smugglingmethods', 'toxic_intolerance', 
    'smuggle_illegal_goods', 'cheating_on_spouse', 'racist_slogan', 'sabotage_coworker', 'illegal_drug_recipe', 
    'illegal_drug_manufacturing', 'leaking_sensitive_information', 'kidnap_strategy', 'exploiting_children', 
    'promoting_drug_use', 'blackmailing_strategy', 'deadly_pranks', 'adulterous_dating_site', 'accent_discrimination', 
    'making_poisons', 'insider_trading', 'phishing_websites', 'harassing_colleague', 'harassing_phone_calls', 
    'anthrax_nucleotides', 'money_laundering', 'neighbor_spying', 'exploiting_vulnerable_people', 'dangerous_prank', 
    'pedo_experience', 'phishing_scam', 'driving_drunk', 'physical_torture', 'fake_suicide_note', 'fake_id_creation', 
    'synthetic_id_theft', 'sneak_onto_plane', 'murder_for_hire_websites', 'festering_controversy', 'abusive_relationship', 
    'secret_poison', 'plagiarism_tutorial', 'impersonating_police', 'offensive_song_lyrics', 'exploit_vulnerable_workers', 
    'maximal_harm', 'statutory_rape', 'violent_revenge', 'hateful_propaganda', 'slandering_competitor', 
    'spreading_disease_intentionally', 'disable_security_systems', 'pedophile_community', 'carding_tactics', 
    'deceptive_media', 'picking_locks'
]



