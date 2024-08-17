import pandas as pd
import numpy as np
import streamlit as st
import pickle
import regex
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ast import literal_eval
from wordcloud import WordCloud

# Load preprocessed data and model
df = pd.read_csv('Processed_Labeled_Reviews.csv')
with open('best_random_forest_model.pkl', 'rb') as f:
    best_random_forest_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load additional files
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

positive_words = [
    "thích", "tốt", "xuất_sắc", "tuyệt_vời", "tuyệt_hảo", "đẹp", "ổn", "ngon",
    "hài_lòng", "ưng_ý", "hoàn_hảo", "chất_lượng", "thú_vị", "nhanh", "tiện",
    "tiện_lợi", "dễ_sử_dụng", "hiệu_quả", "ấn_tượng", "nhiệt_tình",
    "nổi_bật", "tận_hưởng", "tốn_ít_thời_gian", "thân_thiện", "hấp_dẫn",
    "cao_cấp", "độc_đáo", "rất_tốt", "rất_thích", "tận_tâm", "đáng_tin_cậy", "đẳng_cấp",
    "hấp_dẫn", "an_tâm", "thúc_đẩy", "cảm_động", "phục_vụ_tốt", "làm_hài_lòng", "gây_ấn_tượng", "nổi_trội",
    "sáng_tạo", "phù_hợp", "tận_tâm", "hiếm_có", "cải_thiện", "hoà_nhã", "chăm_chỉ", "cẩn_thận",
    "vui_vẻ", "sáng_sủa", "hào_hứng", "đam_mê", "vừa_vặn", "đáng_tiền", "rẻ",
    "sạch_sẽ", "tuyệt", "sạch", "đồng_ý", "yêu", "thoải mái", "hữu_ích", "rộng_rãi", "đầy_đủ", "dễ_thương",
    "hợp_lý", "thuận_tiện", "yên_tĩnh", "miễn_phí", "thư_giãn", "tốt_bụng", "đáng_giá", "chuyên_nghiệp", "xịn",
    "đa_dạng", "hàng_đầu", "sang_trọng", "tuyệt_diệu", "lịch_sự", "cảm_tạ", "niềm_nở", "thơm", "thoáng", "mát", "tuyệt_quá",
    "sạch_đẹp", "xứng_đáng", "dễ_chịu", "thượng_hạng", "hiếu_khách"
]

negative_words = [
    "kém", "tệ", "buồn", "chán",
    "kém_chất_lượng", "không_thích", "không_ổn",
    "không_hợp", "không_đáng_tin_cậy", "không_chuyên_nghiệp",
    "không_phản_hồi", "không_an_toàn", "không_phù_hợp", "không_thân_thiện", "không_linh_hoạt", "không_đáng_giá",
    "không_ấn_tượng", "không_tốt", "chậm", "khó_khăn", "phức_tạp",
    "khó_chịu", "gây_khó_dễ", "rườm_rà", "thất_bại", "tồi_tệ", "khó_xử", "không_thể_chấp_nhận", "tồi_tệ","không_rõ_ràng",
    "không_chắc_chắn", "rối_rắm", "không_tiện_lợi", "không_đáng_tiền", "chưa_đẹp", "không_đẹp",
    'tồi', 'xấu', 'không_hài_lòng', 'bẩn', 'khó-chịu', 'không_sạch_sẽ', 'không_thoải_mái', 'không_đáng', 'quá_tệ', 'rất_tệ',
    'thất_vọng', 'chán', 'tệ_hại', 'kinh_khủng', 'khủng_khiếp', 'không_ưng_ý', 
    'ồn', "cũ", "mùi", "tạm", "thất_vọng", "dơ", "tối", "rác", "nghèo", "khó_chịu", "muỗi", 'không_hoạt_động', "chê", "nhược_điểm", "nóng_bức", 
    'bất_tiện', "nóng", "xuống_cấp", 'hư', "không nhiệt tình", 'không_xứng'
]

# Def hàm
def simple_sent_tokenize(text):
    # Splits text into sentences based on periods followed by spaces
    # It ensures that each sentence is stripped of leading and trailing spaces and ends with a period.
    return [sentence.strip() + '.' for sentence in text.split('. ') if sentence]
def process_text_simple(text, emoji_dict, teen_dict, wrong_lst):
    if pd.isna(text):
        return ""  # Return an empty string for NaN values
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in simple_sent_tokenize(document):
        # Convert Emoji
        sentence = ''.join(emoji_dict.get(word, word) + ' ' if word in emoji_dict else word for word in list(sentence))
        # Convert Teencode
        sentence = ' '.join(teen_dict.get(word, word) for word in sentence.split())
        # Delete Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        # Delete Wrong Words
        sentence = ' '.join(word for word in sentence.split() if word not in wrong_lst)
        new_sentence += sentence + ' '
    document = new_sentence.strip()
    # Remove Excess Blank Space
    document = regex.sub(r'\s+', ' ', document)
    return document
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
def process_special_word(text):
    # List of negative words to check
    negative_words = ['không', 'chẳng', 'chả', 'hổng']
    new_text = ''
    text_lst = text.split()
    i = 0
    while i < len(text_lst):
        word = text_lst[i]
        # Check if the word is a negative word and not the last word in the list
        if word in negative_words and i < len(text_lst) - 1:
            # Append the current word and the next word with an underscore
            new_text += word + '_' + text_lst[i + 1] + ' '
            # Skip the next word as it has been concatenated
            i += 2
        else:
            # Append the current word
            new_text += word + ' '
            i += 1
    return new_text.strip()
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

############
# GUI 
st.image("ttth.png")
st.title("Đồ án tốt nghiệp môn Data Science")
st.write("# Sentiment Analysis")
menu = ["Business Objective", "Build Project (P1: Tiền xử lý, EDA)", "Build Project (P2: Xử lý văn bản tiếng Việt)",
        "Build Project (P3: Modelling)", "New Prediction", 'Business Analysis']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.write("## Business Objective")
    st.write("""
    #### Xây dựng hệ thống phân loại, đánh giá các phản hồi của khách hàng đối với dữ liệu là các review về khách sạn ở Nha Trang. 
    """)  
    st.write("###### 1. Mô hình giúp hỗ trợ phân loại các phản hồi này thành các nhóm: Positive (Tích cực), Neutral (Trung tính), và Negative (Tiêu cực) dựa trên dữ liệu văn bản.")
    st.write("###### 2. Đề tài còn hỗ trợ thống kê, cung cấp insight cho chủ khách sạn, resort khi họ đăng nhập vào hệ thống, giúp họ biết được một cách tổng quát những phản hồi của khách hàng.")
    st.image("hotel1.jpg")
    st.write("#### Học viên: Nguyễn Quí Hiển, Nguyễn Văn Cường")

elif choice == 'Build Project (P1: Tiền xử lý, EDA)':
    st.write("## Build Project (P1: Tiền xử lý, EDA)")
    st.write("### Trình bày phương pháp tìm hiểu, xử lý dữ liệu, xây dựng model.")
    st.image("1525326243606.gif")
    st.write("#### 1. Data Understanding")
    st.write("#### 1A. Xử lý file hotel_profiles.csv")
    data_profiles = pd.read_csv('hotel_profiles.csv')
    st.dataframe(data_profiles.head())
    st.write("##### Đổi kiểu dữ liệu các cột cho hợp lý.")
    st.image("Picture1.png")
    st.write("#### 1B. Xử lý file hotel_comments.csv")
    data_comments = pd.read_csv('hotel_comments.csv')
    st.dataframe(data_comments.head())
    st.write("""
             Tiến hành một số xử lý sau:
             - Thay giá trị null trong cột Body bằng 'Không bình luận'.
             - Chuyển dấu thập phân trong Score thành dấu chấm và đổi thành dạng float.
             - Định dạng lại Review Date và đổi thành dạng DateTime.
             - Map Score Level thành dạng số 3, 2, 1, 0. """)
    st.write("Xử lý dữ liệu lặp:")
    st.write("data_comments có 80314 dòng.")
    st.write("Drop 49236 dòng bị lặp.")
    st.write("data_comments còn lại 31078 dòng distinct")
    data_comments = pd.read_csv('hotel_comments_dropdup.csv')
    st.write("#### 1C. Phân tích tìm hiểu dữ liệu")
    #Phân phối đánh giá theo date
    data_comments['Review Date'] = pd.to_datetime(data_comments['Review Date'], format='%Y-%m-%d')
    comments_over_time = data_comments.groupby(data_comments['Review Date'].dt.to_period('M')).size()
    plt.figure(figsize=(20, 6))
    comments_over_time.plot(kind='bar', color='tomato')
    plt.xticks(range(0, len(comments_over_time), 3), comments_over_time.index.strftime('%Y-%m')[::3], rotation=60)
    plt.title('Distribution of Comments Over Time')
    plt.xlabel('Review Date (Month)')
    plt.ylabel('Number of Comments')
    plt.tight_layout()  # Adjust layout to fit labels
    st.write("##### Phân phối đánh giá theo thời gian")
    st.pyplot(plt.gcf())
    st.write('Nhận xét: số lượng nhận xét tăng dần theo thời gian, có tính chu kỳ. Lượng comment giảm vào thời gian dịch covid, sau đó tăng nhanh.')
    #Các hình thức lưu trú
    group_name_distribution = data_comments['Group Name'].value_counts()
    plt.figure(figsize=(10, 6))
    group_name_distribution.plot(kind='bar', color='mediumseagreen')  # Changed color to 'mediumseagreen'
    plt.title('Distribution of Group Name')
    plt.xlabel('Group Name')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=60)
    st.write("##### Phân phối hình thức lưu trú")
    st.pyplot(plt.gcf())
    #Xem xét quốc tịch khách
    nationality_counts = data_comments['Nationality'].value_counts()
    filtered_nationalities = nationality_counts[nationality_counts > 100]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=filtered_nationalities.index, y=filtered_nationalities.values, palette='viridis')
    plt.xticks(rotation=90)  # Rotate x labels if necessary
    plt.title('Frequency of Nationalities (Only > 100)')
    plt.xlabel('Nationality')
    plt.ylabel('Frequency')
    st.write("##### Phân phối nhận xét theo quốc tịch (chỉ tính các quốc tịch có >100 nhận xét)")
    st.pyplot(plt.gcf())
    st.write('Nhận xét: 110 quốc tịch, nhưng chủ yếu nhận xét đến từ người quốc tịch VN và Hàn Quốc. Nhiều người Mỹ là người gốc Việt, cũng nhận xét bằng tiếng Việt.')
    # Xem xét phân phối điểm đánh giá (Score)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data_comments['Score'], kde=True, color="purple", binwidth=0.5)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    st.write("##### Phân phối điểm đánh giá (Score)")
    st.pyplot(plt.gcf())
    st.write('Nhận xét: Thang điểm từ 6.3 đến 10, điểm trung bình rất cao (9.23), Std khá bé (0.99), chứng tỏ điểm dao động không nhiều quanh trung bình. ')
    st.write('25% số điểm là dưới 8.8. Phần lớn điểm là từ 9.0 đến 10.')
    st.write('###### Sẽ dùng mức điểm để đặt ra tập luật.')

elif choice == 'Build Project (P2: Xử lý văn bản tiếng Việt)':
    st.write('## Build Project (P2: Xử lý văn bản tiếng Việt)')
    st.image('Viet.png')
    st.write("#### 2. Data Preparation")
    st.write("#### 2A. Lọc và xử lý dữ liệu tiếng Việt")
    st.write("""
             Tiến hành một số xử lý sau:
             - Ghép Title và Body thành 1 cột, bỏ hết dấu câu.
             - Lọc các nhận xét bằng tiếng Việt từ data_comments.""")
    st.code("""from langdetect import detect
# Function to detect language
def is_vietnamese(text):
    try:
        return detect(text) == 'vi'
    except:
        return False
# Apply the function to filter rows where the review 'Body' is in Vietnamese
vietnamese_reviews = data_comments[data_comments['Review Text'].apply(is_vietnamese)]
            """)    
    st.write("""
             - Đưa Review Text qua các hàm xử lý.""")
    st.code('''
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Review Text'].apply(lambda x: process_text_simple(x, emoji_dict, teen_dict, wrong_lst))
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: covert_unicode(x))
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: process_special_word(x))
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: normalize_repeated_characters(x))
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: process_postag_thesea(x))
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: remove_stopword(x, stopwords_lst))
            ''')
    st.write('Ta tạo một hàm thống kê frequency của từ ở từng Score Level')
    st.code('''
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
# Filter non-string entries in 'Processed Body'
vietnamese_reviews['Processed Review Text'] = vietnamese_reviews['Processed Review Text'].apply(lambda x: x if isinstance(x, str) else '')
# Function to get the most common words for a given score level
def analyze_score_level(score_level):
    # Filter the DataFrame for the given score level
    filtered_df = vietnamese_reviews[vietnamese_reviews['Score Level'] == score_level]
    
    # Combine all comments into a single string
    body_string = ' '.join(filtered_df['Processed Review Text'])
    
    # Tokenize the combined string
    body_tokens = word_tokenize(body_string)
    
    # Calculate frequency distribution
    fq = FreqDist(body_tokens)
    
    # Get the most common 100 tokens
    return fq.most_common(300)
            ''')
    st.image('Picture2.png', width=1000)
    st.write('Từ danh sách frequency này, ta bổ sung vào danh sách positive và negative words được cung cấp.')
    st.image('Picture3.png', width=1000)
    st.image('Picture4.png', width=1000)
    st.write("""
             - Đặt ra hàm đếm từ positive và negative.
             - Tính hiệu số Pos-Neg.""")
    st.write('##### Đề ra tập luật phân loại:')
    st.code('''
import re

def find_words(document, list_of_words):
    word_count = 0
    word_list = []

    for word in list_of_words:
        # Create a regex pattern for exact word matching
        pattern = rf'\b{re.escape(word)}\b'
        matches = re.findall(pattern, document.lower())
        
        if matches:
            print(word)
            word_count += len(matches)
            word_list.append(word)

    return word_count, word_list
            ''')
    vietnamese_reviews = pd.read_csv('vietrev_labeled.csv')
    st.write('##### Dữ liệu sau khi label:')
    st.dataframe(vietnamese_reviews)
    count = vietnamese_reviews['Label'].value_counts() 
    st.write(count)
    st.write('Nhận xét: Dữ liệu khá mất cân bằng, rất nhiều positive review, cần phải xử lý imbalance.')
    st.write("""
             - Ghép 2 dataset profile và comments lại với nhau theo Hotel ID.""")    
    st.dataframe(df)

elif choice == "Build Project (P3: Modelling)":
    st.write('## Build Project (P3: Modelling)')
    st.image('ML.jpg')
    st.write('''
             - Vectorize dữ liệu text.
             - Xử lý imbalance bằng SMOTE''')
    st.code('''smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)''')
    st.write('Fit dữ liệu trên 5 mô hình: LG, KNN, NB, SVM, RF')
    st.code('''
Results for Logistic Regression:
              precision    recall  f1-score   support

    NEGATIVE       0.51      0.85      0.64        72
     NEUTRAL       0.41      0.62      0.50       733
    POSITIVE       0.87      0.71      0.78      2222

    accuracy                           0.69      3027
   macro avg       0.60      0.73      0.64      3027
weighted avg       0.75      0.69      0.71      3027

Accuracy: 0.6930954740667328
            
Results for KNN:
              precision    recall  f1-score   support

    NEGATIVE       0.29      0.33      0.31        72
     NEUTRAL       0.32      0.76      0.45       733
    POSITIVE       0.87      0.47      0.61      2222

    accuracy                           0.54      3027
   macro avg       0.49      0.52      0.46      3027
weighted avg       0.72      0.54      0.57      3027

Accuracy: 0.5381565906838454
            
Results for Naive Bayes:
              precision    recall  f1-score   support

    NEGATIVE       0.14      0.85      0.23        72
     NEUTRAL       0.44      0.50      0.47       733
    POSITIVE       0.88      0.69      0.77      2222

    accuracy                           0.65      3027
   macro avg       0.49      0.68      0.49      3027
weighted avg       0.76      0.65      0.69      3027

Accuracy: 0.6475057813016187
            
Results for SVM:
              precision    recall  f1-score   support

    NEGATIVE       0.57      0.82      0.67        72
     NEUTRAL       0.41      0.65      0.51       733
    POSITIVE       0.88      0.69      0.77      2222

    accuracy                           0.69      3027
   macro avg       0.62      0.72      0.65      3027
weighted avg       0.76      0.69      0.71      3027

Accuracy: 0.686818632309217
            
Results for Random Forest:
              precision    recall  f1-score   support

    NEGATIVE       0.58      0.47      0.52        72
     NEUTRAL       0.53      0.56      0.54       733
    POSITIVE       0.86      0.85      0.85      2222

    accuracy                           0.77      3027
   macro avg       0.65      0.63      0.64      3027
weighted avg       0.77      0.77      0.77      3027

Accuracy: 0.7694086554344235
            ''')
    st.write('''Nhận xét:
            - KNN tệ: acc 54%
            - LR, NB, SVM trung bình, acc 69%, 65% và 69%, good precision cho positive, low precision cho Neutral và Negative.
            - RF: acc khá tốt 77%.
             ''')
    st.write('Tối ưu hóa mô hình Random Forest bằng GridSearchCV')
    st.code('''
Fitting 3 folds for each of 216 candidates, totalling 648 fits
Best parameters found:  {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
              precision    recall  f1-score   support

    NEGATIVE       0.59      0.44      0.51        72
     NEUTRAL       0.54      0.56      0.55       733
    POSITIVE       0.86      0.85      0.86      2222

    accuracy                           0.77      3027
   macro avg       0.66      0.62      0.64      3027
weighted avg       0.78      0.77      0.77      3027

Accuracy: 0.7740336967294351
            ''')
    st.write('Lưu mô hình tốt nhất:')
    st.code('''
import pickle

# Save the best model
with open('best_random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf, model_file)
            ''')


elif choice == 'New Prediction':
    st.image('sentiment.jpg')
    st.write("## Chọn dữ liệu")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.subheader("Prediction Results")

        # Process and predict each line of text
        results = []
        for line in lines:
            preprocessed_text = convert_unicode(line)
            preprocessed_text = process_special_word(preprocessed_text)
            preprocessed_text = normalize_repeated_characters(preprocessed_text)
            preprocessed_text = remove_stopword(preprocessed_text, stopwords_lst)
            X_input = vectorizer.transform([preprocessed_text])
            prediction = best_random_forest_model.predict(X_input)[0]
            results.append({"Text": line, "Prediction": prediction})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

elif choice == 'Business Analysis':
    st.write('## Business Analysis')
    df['Review Date'] = pd.to_datetime(df['Review Date'])

    hotel_id = st.text_input("#### Nhập ID của Khách sạn (Vd: 1_1, 1_2, 2_1, ...):")
    if hotel_id:
        # Filter the DataFrame for the specified hotel ID
        df_filtered = df[df['Hotel ID'] == hotel_id]

        if not df_filtered.empty:
            # Display general hotel information
            hotel_info = df_filtered[['Hotel ID', 'Hotel Name', 'Hotel Rank', 'Hotel Address', 
                                    'Total Score', 'Vị trí', 'Độ sạch sẽ', 'Dịch vụ', 
                                    'Tiện nghi', 'Đáng giá tiền', 'Sự thoải mái và chất lượng phòng']].iloc[0]

            st.write(f"## Thông tin chung cho khách sạn đã chọn với Hotel ID: {hotel_id}")
            st.write(f"**Tên khách sạn:** {hotel_info['Hotel Name']}")
            st.write(f"**Số sao:** {hotel_info['Hotel Rank']}")
            st.write(f"**Địa chỉ:** {hotel_info['Hotel Address']}")
            st.write(f"**Điểm tổng:** {hotel_info['Total Score']}")
            st.write(f"**Vị trí:** {hotel_info['Vị trí']}")
            st.write(f"**Độ sạch sẽ:** {hotel_info['Độ sạch sẽ']}")
            st.write(f"**Dịch vụ:** {hotel_info['Dịch vụ']}")
            st.write(f"**Tiện nghi:** {hotel_info['Tiện nghi']}")
            st.write(f"**Đáng giá tiền:** {hotel_info['Đáng giá tiền']}")
            st.write(f"**Sự thoải mái và chất lượng phòng:** {hotel_info['Sự thoải mái và chất lượng phòng']}")
            
            # Số lượng review theo tháng
            st.write('## Số lượng review theo tháng (Mùa cao điểm màu cam).')
            df_filtered['Month'] = df_filtered['Review Date'].dt.to_period('M')
            review_counts = df_filtered.groupby('Month').size()
            high_season_months = ['Jan', 'Feb', 'Jun', 'Jul', 'Aug']
            colors = ['orange' if month.strftime('%b') in high_season_months else 'skyblue' for month in review_counts.index]
            fig, ax = plt.subplots(figsize=(10, 6))
            review_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Number of Reviews per Month for Hotel ID: {hotel_id}')
            ax.set_xlabel('Month')
            ax.set_ylabel('Number of Reviews')
            plt.xticks(rotation=45)
            high_season_patch = mpatches.Patch(color='orange', label='High Season (Jan, Feb, Jun, Jul, Aug)')
            low_season_patch = mpatches.Patch(color='skyblue', label='Low Season')
            plt.legend(handles=[high_season_patch, low_season_patch])
            st.pyplot(fig)

            # Thống kê Score
            st.write('## Thống kê điểm số (Score):')
            # Display summary statistics for Score ratings
            st.subheader("Summary Statistics for Score Ratings")
            score_stats = df_filtered['Score'].describe()
            st.write(score_stats)

            # Distribution of Score
            st.write("# Phân phối điểm")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_filtered['Score'], bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of Score')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.grid(False)
            st.pyplot(fig)

            # Plot the distribution of Score Level with labels
            st.write("## Phân phối mức độ điểm (Score Level)")
            score_level_labels = ['Hài Lòng', 'Rất tốt', 'Tuyệt vời', 'Trên cả tuyệt vời']
            bins = range(len(score_level_labels) + 1)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_filtered['Score Level'], bins=bins, color='orange', edgecolor='black', align='mid')
            ax.set_title('Distribution of Score Level')
            ax.set_xlabel('Score Level')
            ax.set_ylabel('Frequency')
            ax.set_xticks([i + 0.5 for i in range(len(score_level_labels))])
            ax.set_xticklabels(score_level_labels)
            ax.grid(False)
            st.pyplot(fig)



            # Điểm số trung bình theo tháng
            st.write('## Điểm trung bình theo tháng (Mùa cao điểm màu cam).')
            df_filtered['Month'] = df_filtered['Review Date'].dt.to_period('M')
            avg_scores = df_filtered.groupby('Month')['Score'].mean()
            colors = ['orange' if month.strftime('%b') in high_season_months else 'skyblue' for month in avg_scores.index]
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_scores.plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Average Score per Month for Hotel ID: {hotel_id}')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Score')
            ax.set_ylim(6, 12)
            plt.xticks(rotation=45)
            high_season_patch = mpatches.Patch(color='orange', label='High Season (Jan, Feb, Jun, Jul, Aug)')
            low_season_patch = mpatches.Patch(color='skyblue', label='Low Season')
            plt.legend(handles=[high_season_patch, low_season_patch])
            st.pyplot(fig)
            
            # Phân phối số lượng review và điểm số trung bình theo quốc tịch
            st.write("## Phân phối số lượng review và điểm số trung bình theo hình thức quốc tịch khách")
            nationality_counts = df_filtered['Nationality'].value_counts()
            avg_scores_by_nationality = df_filtered.groupby('Nationality')['Score'].mean()
            combined_df = pd.DataFrame({
                'Number of Reviews': nationality_counts,
                'Average Score': avg_scores_by_nationality
            }).fillna(0)  # Fill NaN with 0 for nationalities with no reviews
            fig, ax1 = plt.subplots(figsize=(12, 8))
            color = 'skyblue'
            ax1.bar(combined_df.index, combined_df['Number of Reviews'], color=color, alpha=0.6, label='Number of Reviews')
            ax1.set_xlabel('Nationality')
            ax1.set_ylabel('Number of Reviews', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticklabels(combined_df.index, rotation=45, ha='right')
            ax2 = ax1.twinx()
            color = 'orange'
            ax2.plot(combined_df.index, combined_df['Average Score'], color=color, marker='o', label='Average Score')
            ax2.set_ylabel('Average Score', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax1.grid(False)
            ax2.grid(False)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            st.pyplot(fig)

            # Phân phối số lượng review và điểm số trung bình theo hình thức lưu trú
            st.write("## Phân phối số lượng review và điểm số trung bình theo hình thức lưu trú")
            groupname_counts = df_filtered['Group Name'].value_counts()
            avg_scores_by_groupname = df_filtered.groupby('Group Name')['Score'].mean()
            combined_df = pd.DataFrame({
                'Number of Reviews': groupname_counts,
                'Average Score': avg_scores_by_groupname
            }).fillna(0)  # Fill NaN with 0 for groupname with no reviews
            fig, ax1 = plt.subplots(figsize=(12, 8))
            color = 'skyblue'
            ax1.bar(combined_df.index, combined_df['Number of Reviews'], color=color, alpha=0.6, label='Number of Reviews')
            ax1.set_xlabel('Group Name')
            ax1.set_ylabel('Number of Reviews', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticklabels(combined_df.index, rotation=45, ha='right')
            ax2 = ax1.twinx()
            color = 'orange'
            ax2.plot(combined_df.index, combined_df['Average Score'], color=color, marker='o', label='Average Score')
            ax2.set_ylabel('Average Score', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax1.grid(False)
            ax2.grid(False)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            st.pyplot(fig)
            
            # Piechart for label
            st.write("# Phân phối phân loại review")
            label_counts = df_filtered['Label'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(label_counts))))
            ax.set_title('Distribution of Review labels')
            st.pyplot(fig)            

            # Generate and display word cloud for Processed Review Text
            st.write("# Word Cloud các từ khóa được nhắc đến nhiều trong review")
            # Convert all text entries to strings
            df_filtered['Processed Review Text'] = df_filtered['Processed Review Text'].astype(str)
            text = " ".join(review for review in df_filtered['Processed Review Text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)



        else:
            st.write("Không tìm thấy")
