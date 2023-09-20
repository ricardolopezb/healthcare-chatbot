import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import difflib
import ast
import pandas as pd  # Import pandas library
import spacy
import openai
import os
from dotenv import load_dotenv

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

symptoms_in_spanish = ['picazon', 'erupcion_cutanea', 'erupciones_cutaneas_nodales', 'estornudos_continuos', 'escalofrios', 'escalofrios', 'dolor_en_las_articulaciones', 'dolor_de_estomago', 'acidez', 'ulceras_en_la_lengua', 'perdida_de_musculo', 'vomitos', 'miccion_ardiente', 'miccion_con_manchas', 'fatiga', 'aumento_de_peso', 'ansiedad', 'manos_y_pies_frios', 'cambios_de_humor', 'perdida_de_peso', 'inquietud', 'letargo', 'manchas_en_la_garganta', 'niveles_irregulares_de_azucar', 'tos', 'fiebre_alta', 'ojos_hundidos', 'falta_de_aliento', 'sudoracion', 'desHidratacion', 'indigestion', 'dolor_de_cabeza', 'piel_amarillenta', 'orina_oscura', 'nauseas', 'perdida_de_apetito', 'dolor_detrás_de_los_ojos', 'dolor_de_espalda', 'estreñimiento', 'dolor_abdominal', 'diarrea', 'fiebre_leve', 'orina_amarilla', 'ojos_amarillos', 'fracaso_hepatico_agudo', 'sobrecarga_de_liquido', 'hinchazon_del_estomago', 'ganglios_linfaticos_hinchados', 'malestar', 'vision_borrosa_y_distorsionada', 'flema', 'irritacion_de_garganta', 'ojos_rojos', 'presion_sinus', 'nariz_corriendo', 'congestion', 'dolor_en_el_pecho', 'debilidad_en_las_extremidades', 'ritmo_cardiaco_rapido', 'dolor_durante_la_evacuacion_intestinal', 'dolor_en_la_region_anal', 'heces_sangrantes', 'irritacion_en_el_ano', 'dolor_de_cuello', 'mareos', 'calambres', 'moretones', 'obesidad', 'piernas_hinchadas', 'vasos_sanguineos_hinchados', 'cara_y_ojos_hinchados', 'tiroides_agrandada', 'unas_quebradizas', 'extremidades_hinchadas', 'hambre_excesiva', 'contactos_extra_matrimoniales', 'labios_secos_y_tingling', 'habla_arrastrada', 'dolor_de_rodilla', 'dolor_en_la_cadera', 'debilidad_muscular', 'cuello_rigido', 'articulaciones_hinchadas', 'rigidez_en_el_movimiento', 'movimientos_giratorios', 'perdida_de_equilibrio', 'inestabilidad', 'debilidad_en_un_lado_del_cuerpo', 'perdida_del_olfato', 'malestar_en_la_vejiga', 'olor_fetido_de_la_orina', 'sensacion_continua_de_orina', 'paso_de_gases', 'picazon_interna', 'apariencia_toxica_(tifos)', 'depresion', 'irritabilidad', 'dolor_muscular', 'sentido_alterado', 'manchas_rojas_en_el_cuerpo', 'dolor_abdominal', 'manchas_discromaticas', 'lagrimeo', 'aumento_del_apetito', 'poliuria', 'antecedentes_familiares', 'esputo_mucoso', 'esputo_oxidado', 'falta_de_concentracion', 'trastornos_visuales', 'recepcion_de_transfusion_de_sangre', 'recepcion_de_inyecciones_no_esteriles', 'coma', 'hemorragia_gastrica', 'distension_abdominal', 'historia_de_consumo_de_alcohol', 'sobrecarga_de_liquido.1', 'sangre_en_el_esputo', 'venas_destacadas_en_la_pantorrilla', 'palpitaciones', 'dolor_al_caminar', 'espinillas_llenas_de_pus', 'puntos_negros', 'descamacion_de_la_piel', 'polvo_como_la_formacion_de_plata', 'pequenas_muescas_en_las_unas', 'unas_inflamadas', 'ampolla', 'llagas_rojas_alrededor_de_la_nariz', 'costra_amarilla_supurante']

nlp = spacy.load("es_core_news_sm")
#nlp = spacy.load("en_core_web_sm")
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_words_from_file(filename):
    try:
        with open(filename, 'r') as file:
            word_list = file.read().split()
        return word_list
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []

medical_stopwords = read_words_from_file("medical_stopwords.txt")

def save_text_to_file(text, filename):
    with open(filename, 'a') as file:
        file.write(text+"\n\n")

def parse_string_to_list(input_str):
    try:
        parsed_list = ast.literal_eval(input_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("Input is not a valid list representation.")
    except (SyntaxError, ValueError):
        raise ValueError("Input is not a valid list representation.")

def send_openai_request(text_input):
    df_train = pd.read_csv('./Data/Training.csv', delimiter=',') # try to optimize this
    vocab = df_train.columns.tolist()

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "system",
            "content": f'''
                Given the following list of symptoms:
                {vocab}
                If someone says "{text_input}", convert it to english in case it isn't,
                what symptoms is it most likely to reference?
                Answer it as a system, and the output is a list of plain strings, just the names of the symptoms. For example, return just: ["headache", "nausea"] and nothing more than that. If you don't send it to me in that format i will kill myself swear to God.
            '''
        }
    ]
    )
    # TODO check if the returned response is of format [...]
    return parse_string_to_list(response.choices[0].message.content)

# This removes words that are in the stop words
def word_extractor(sentence, stopwords):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if not token in stopwords]
    return tokens


# This only extracts nouns in the sentence
def noun_token_extractor(sentence):
    doc = nlp(sentence)
    for tok in doc:
        print(f'{tok} - {tok.pos_}')
    noun_tokens = [str(token) for token in doc if token.pos_ == "NOUN"]
    return noun_tokens

def symptoms(symptoms):
    final_symptoms = []
    final_symptoms_flat = []
    df_train = pd.read_csv('./Data/Training.csv', delimiter=',')
    vocab = df_train.columns.tolist() + symptoms_in_spanish

    for symptom in symptoms:
        final_symptoms.append(difflib.get_close_matches(symptom, vocab, cutoff=0.6))
    for sublist in final_symptoms:
        for item in sublist:
            final_symptoms_flat.append(item)

    return list(final_symptoms_flat)

'''
    A veces responde solo el array para parsear, pero otras responde una respuesta armada 

    Ej: "content": "The symptoms mentioned in the Spanish sentence are:\n\n- manchas violeta en mi cuerpo (violet spots on my body)\n- si las apreto me duele (if I press them, they hurt)\n\nBased on these symptoms, the most likely referenced symptoms would be:\n\n- red_spots_over_body (red spots over body)\n- pain_behind_the_eyes (pain behind the eyes)"},

    Se arregla con mejor prompting?
'''

#send_openai_request("me duele la espalda, el cuello y estoy muy triste") # funciona bien
#send_openai_request("veo manchas moradas en mi cuerpo y si las apreto me duele") # esta mal, deberia incluir bruising
#send_openai_request("no me puedo levantar de la cama, siento que no tengo energia") # esta bien

# description = "feeling my head hurt"
# a = symptoms(noun_token_extractor(description))

# print(a)

#description = "no me puedo levantar de la cama, siento que no tengo energia"




descriptions = [
    "My head hurts",
    "I am feeling nausea",
    "I can't get out of my bed, I don't have any energy",
    "My back and neck hurt and I am feeling very sad",
    "My head and neck hurt",
    "I am feeling a gooey thing in my throat that is making me cough",

    "Me duele la cabeza",
    "Me siento con náuseas",
    "No puedo salir de mi cama, no tengo energía",
    "Me duelen la espalda y el cuello y me siento muy triste",
    "Me duelen la cabeza y el cuello",
    "Siento algo pegajoso en la garganta que me hace toser"
]




# for description in descriptions:
#     print(f"[LOG] - Tokenizing {description}")
#     result1 = ("openai", send_openai_request(description))
#     result2 = ("regular stopwords extractor", symptoms(word_extractor(description, stopwords.words())))
#     result3 = ("medical stopwords extractor", symptoms(word_extractor(description, medical_stopwords)))
#     result4 = ("noun extractor", symptoms(noun_token_extractor(description)))
#
#     results = [result1, result2, result3, result4]
#
#     string_to_save = f"DESCRIPTION: {description}\n\n"
#
#     for result in results:
#         string_to_save = string_to_save + f"{result[0]}: {result[1]}\n"
#
#     save_text_to_file(string_to_save, "nlp_test_results.txt")
#
# print("[LOG] - Finished!")