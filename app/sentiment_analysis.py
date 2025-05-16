import re
import math
from collections import defaultdict, Counter # Counter để đếm tần suất
import string

# Bỏ dấu tiếng Việt (hàm này bạn có thể tìm và tích hợp, hoặc dùng cách đơn giản hơn)
# Hoặc tạm thời chấp nhận việc so sánh có dấu.
# Nếu không có hàm bỏ dấu tốt, có thể dẫn đến "tốt" và "tôt" là 2 từ khác nhau.
# Dưới đây là một hàm đơn giản để loại bỏ dấu cơ bản (không hoàn hảo)
def remove_vietnamese_diacritics(text):
    """
    Simplified function to remove Vietnamese diacritics.
    This is a very basic approach and might not cover all cases or be perfect.
    For robust removal, a dedicated library or a more comprehensive mapping is needed.
    """
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in text:
        try:
            s += s0[s1.index(c)]
        except ValueError: # Character not in s1, keep it as is
            s += c
    return s

# Từ dừng - đã được tối ưu một chút, nhưng vẫn cần mở rộng
VIETNAMESE_STOPWORDS = set([
    'va', 'la', 'ma', 'thi', 'o', 'tai', 'trong', 'ngoai', 'tren', 'duoi',
    'cua', 'voi', 'cho', 'de', 'khi', 'sau', 'truoc', 'nhu', 'nay', 'do', 'kia',
    'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
    'rang', 'rat', 'nhung', 'neu', 'hoac', 'cung', 'da', 'se', 'dang', 'duoc',
    'bi', 'boi', 'vi', 'nen', 'khong', 'co', 'chua', 'bao gio', 'luon',
    'moi', 'moi', 'tung', 'tat ca', 'chi', 'rieng', 'khac', 'anh', 'em',
    'chi', 'ong', 'ba', 'co', 'chu', 'bac', 'minh', 'ban', 'toi', 'chung ta',
    'chung toi', 'ho', 'ay', 'oi', 'a', 'ạ', 'di', 'lam', 'an', 've',
    'bang', 'cai', 'cac', 'qua', 'thay', 'qua', # loại bỏ từ "quá" vì nó có thể xuất hiện cả trong tích cực (quá tốt) và tiêu cực (quá tệ)
    # Giữ lại các từ thường dùng trong tiếng Anh
    'the', 'is', 'in', 'it', 'of', 'to', 'and', 'a', 'with', 'for', 'on', 'was', 'were', 'be', 'been'
])


# Danh sách từ khóa phủ định và cách kết hợp (đơn giản)
NEGATIVE_PREFIXES = {
    "khong": "not_",  # ví dụ: không tốt -> not_tot
    "chua": "notyet_", # ví dụ: chưa hài lòng -> notyet_hailong
}

def preprocess_text(text, remove_diacritics=True):
    if remove_diacritics:
        text = remove_vietnamese_diacritics(text) # Bỏ dấu
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) # Loại bỏ dấu câu
    text = re.sub(r'\d+', ' _number_ ', text) # Thay số bằng token đặc biệt

    tokens = text.split()
    processed_tokens = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        # Xử lý phủ định đơn giản
        is_negated = False
        for prefix, combined_prefix in NEGATIVE_PREFIXES.items():
            if word == prefix and i + 1 < len(tokens):
                next_word = tokens[i+1]
                if next_word not in VIETNAMESE_STOPWORDS: # Chỉ kết hợp nếu từ sau không phải stopword
                    processed_tokens.append(combined_prefix + next_word)
                    i += 1 # Bỏ qua từ tiếp theo vì đã kết hợp
                    is_negated = True
                    break
        if not is_negated:
            if word not in VIETNAMESE_STOPWORDS and len(word) > 1 : # Loại bỏ từ 1 ký tự và stopword
                 if not (len(word) == 2 and word.endswith('a')): # Loại bỏ thêm các từ 2 ký tự kết thúc bằng 'a' nếu cần (thường là tên riêng hoặc lỗi)
                    processed_tokens.append(word)
        i += 1

    return processed_tokens


class NaiveBayesClassifier:
    def __init__(self):
        self.word_class_counts = defaultdict(lambda: defaultdict(int)) # word_class_counts[word][class] = count
        self.class_doc_counts = defaultdict(int) # class_doc_counts[class] = number of docs in this class
        self.vocab = set()
        self.total_documents = 0
        self.labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"] # Định nghĩa sẵn các lớp

        # Từ khóa có trọng số cao cho từng lớp (heuristic)
        self.strong_keywords = {
            "POSITIVE": {"tuyet voi", "xuat sac", "hai long", "tot", "ung y", "dang tien", "chat luong cao", "thich", "dep"},
            "NEGATIVE": {"te", "kem", "xau", "that vong", "chan", "loi", "hong", "phi tien", "can than", "khong nen"},
            # Neutral không cần từ khóa mạnh lắm
        }
        # DỮ LIỆU HUẤN LUYỆN GIẢ LẬP - TĂNG CƯỜNG RẤT NHIỀU
        # Đây là phần quan trọng nhất để cải thiện mô hình "thủ công"
        # Tập trung vào các câu thực tế, bao gồm cả những câu dễ gây nhầm lẫn
        training_data = [
            # POSITIVE
            ("san pham rat tot, dung thich lam", "POSITIVE"),
            ("chat luong tuyet voi, se ung ho tiep", "POSITIVE"),
            ("dep hon mong doi, shop giao hang nhanh", "POSITIVE"),
            ("hoan toan hai long voi san pham nay", "POSITIVE"),
            ("qua xung dang voi gia tien", "POSITIVE"),
            ("may chay em, pin trau", "POSITIVE"),
            ("thiet ke sang trong, bat mat", "POSITIVE"),
            ("mau sac dung nhu hinh", "POSITIVE"),
            ("khong co gi de che", "POSITIVE"),
            ("cuc ky ung y", "POSITIVE"),
            ("nen mua nhe moi nguoi", "POSITIVE"),
            ("dich vu cham soc khach hang tot", "POSITIVE"),
            ("dong goi chac chan, can than", "POSITIVE"),
            ("lan dau mua ma thay ok lam", "POSITIVE"),
            ("minh rat thich san pham nay", "POSITIVE"),
            ("gia hop ly, chat luong on", "POSITIVE"), # "on" có thể hơi trung tính, nhưng câu tổng thể tích cực
            ("san pham dung nhu mo ta cua shop", "POSITIVE"),
            ("mua lan thu hai van rat hai long", "POSITIVE"),
            ("tot ngoai suc tuong tuong", "POSITIVE"),
            ("san pham chinh hang, yen tam su dung", "POSITIVE"),

            # NEGATIVE
            ("san pham qua xau, khong nhu quang cao", "NEGATIVE"),
            ("chat luong kem, moi dung da hong", "NEGATIVE"),
            ("that vong tran tre, khong nen mua", "NEGATIVE"),
            ("hang loi, shop khong giai quyet", "NEGATIVE"),
            ("giao sai mau, sai kich thuoc", "NEGATIVE"),
            ("dung rat chan, hay bi loi vat", "NEGATIVE"),
            ("phi tien mua san pham nay", "NEGATIVE"),
            ("khong tot chut nao, qua te", "NEGATIVE"),
            ("san pham khong nhu mong doi", "NEGATIVE"),
            ("minh khong hai long voi chat luong", "NEGATIVE"),
            ("cam giac nhu bi lua", "NEGATIVE"),
            ("moi nguoi can than khi mua hang cua shop nay", "NEGATIVE"),
            ("mau ma xau hon tren hinh nhieu", "NEGATIVE"),
            ("pin tut nhanh kinh khung", "NEGATIVE"),
            ("san pham co mui kho chiu", "NEGATIVE"),
            ("khong xung dang voi so tien bo ra", "NEGATIVE"),
            ("chat lieu re tien, khong ben", "NEGATIVE"),
            ("shop lam an khong uy tin", "NEGATIVE"),
            ("qua that vong ve san pham nay", "NEGATIVE"), # Cụ thể hóa "sản phẩm"
            ("dien thoai nay te that su", "NEGATIVE"),
            ("may tinh chay rat cham va lag", "NEGATIVE"),
            ("khong hai long ti nao", "NEGATIVE"),

            # NEUTRAL
            ("san pham dung tam duoc", "NEUTRAL"),
            ("cung binh thuong, khong co gi noi bat", "NEUTRAL"),
            ("giao hang dung hen", "NEUTRAL"),
            ("mau sac hoi khac so voi anh mot chut", "NEUTRAL"),
            ("gia ca phu hop", "NEUTRAL"),
            ("shop phan hoi tin nhan cham", "NEUTRAL"), # Có thể hơi tiêu cực nhưng tổng thể chưa đủ mạnh
            ("dong goi tam on", "NEUTRAL"),
            ("san pham o muc chap nhan duoc", "NEUTRAL"),
            ("khong tot khong xau", "NEUTRAL"),
            ("can xem xet them", "NEUTRAL"),
            ("hop dung hoi meo mot chut", "NEUTRAL"),
            ("chi tiet hoan thien chua cao lam", "NEUTRAL"),
            ("voi gia nay thi ok", "NEUTRAL"),
            ("nhan duoc hang roi", "NEUTRAL"),
            ("san pham dung nhu hinh anh", "NEUTRAL") # Hơi giống Positive, nhưng chưa đủ mạnh
        ]
        self._train_from_samples(training_data)

    def _add_sample_counts(self, text_tokens, label):
        self.total_documents += 1
        self.class_doc_counts[label] += 1
        for word in text_tokens:
            self.word_class_counts[word][label] += 1
            self.vocab.add(word)

    def _train_from_samples(self, data):
        for text, label in data:
            processed_text = preprocess_text(text)
            self._add_sample_counts(processed_text, label)

    # Hàm train chuẩn, nếu bạn muốn truyền dữ liệu từ bên ngoài vào
    def train(self, data_tuples_list):
        # Reset
        self.word_class_counts = defaultdict(lambda: defaultdict(int))
        self.class_doc_counts = defaultdict(int)
        self.vocab = set()
        self.total_documents = 0
        self._train_from_samples(data_tuples_list)
        print(f"Đã huấn luyện trên {self.total_documents} mẫu. Tổng số từ vựng: {len(self.vocab)}")

    def predict(self, text):
        if not self.vocab or self.total_documents == 0:
            return "NEUTRAL" # Hoặc ném lỗi

        processed_text = preprocess_text(text)
        if not processed_text: # Nếu sau tiền xử lý không còn từ nào
            return "NEUTRAL"

        # Tính P(word) - xác suất từ xuất hiện trong toàn bộ kho ngữ liệu (dùng cho debug nếu cần)
        # word_overall_counts = Counter(word for doc_tokens in all_doc_tokens_list for word in doc_tokens)

        log_scores = defaultdict(float)

        for label in self.labels:
            if self.class_doc_counts[label] == 0: continue # Bỏ qua lớp không có mẫu

            # P(label) - xác suất tiên nghiệm của lớp
            prob_label = self.class_doc_counts[label] / self.total_documents
            log_scores[label] = math.log(prob_label)

            # Tính tổng số lần xuất hiện của tất cả các từ trong lớp `label`
            total_words_in_class = sum(self.word_class_counts[word][label] for word in self.vocab if label in self.word_class_counts[word])

            # P(word|label) - xác suất có điều kiện
            for word in processed_text:
                # Laplace smoothing (alpha = 1)
                word_count_in_class = self.word_class_counts[word].get(label, 0) + 1
                # Mẫu số: (Tổng số từ trong lớp `label`) + (Kích thước từ vựng)
                denominator = total_words_in_class + len(self.vocab)
                if denominator == 0: continue # Cực hiếm

                prob_word_given_label = word_count_in_class / denominator
                log_scores[label] += math.log(prob_word_given_label)

            # Áp dụng trọng số heuristic cho từ khóa mạnh
            keyword_score_boost = 0
            for kw in self.strong_keywords.get(label, set()):
                # Nếu một từ khóa mạnh xuất hiện, tăng nhẹ score cho lớp đó
                # Việc tăng bao nhiêu là tùy chỉnh
                if kw in processed_text:
                     keyword_score_boost += 0.5 # Tăng điểm một chút cho từ khóa mạnh

            log_scores[label] += keyword_score_boost


        if not log_scores: # Không tính được score nào (rất hiếm nếu có vocab)
            return "NEUTRAL"

        # Lấy nhãn có log score cao nhất
        predicted_label = max(log_scores, key=log_scores.get)

        # (Optional) Xử lý trường hợp điểm quá gần nhau, có thể trả về NEUTRAL
        sorted_scores = sorted(log_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.5: # Ngưỡng tùy chỉnh
             # Nếu điểm cao nhất và điểm thứ hai quá gần nhau, có thể không chắc chắn -> Neutral
             # print(f"Scores too close for '{text[:30]}...', defaulting to NEUTRAL. Scores: {log_scores}")
             # return "NEUTRAL" # Bỏ comment dòng này nếu muốn áp dụng
             pass


        # Debug: In ra điểm số để kiểm tra
        # print(f"Input: '{text}' -> Processed: '{processed_text}' -> Scores: {dict(log_scores)} -> Predicted: {predicted_label}")
        return predicted_label

# Tạo instance classifier để sử dụng
sentiment_classifier = NaiveBayesClassifier()

# Phần Test
if __name__ == "__main__":
    classifier = sentiment_classifier # Đã được huấn luyện với dữ liệu mẫu trong __init__

    test_comments = [
        "san pham qua xau", # Kỳ vọng: NEGATIVE
        "rat tot", # Kỳ vọng: POSITIVE
        "khong dep lam", # Kỳ vọng: NEGATIVE (do "khong_dep") hoặc NEUTRAL
        "chat luong kem", # Kỳ vọng: NEGATIVE
        "tam on", # Kỳ vọng: NEUTRAL
        "moi thu deu tuyet voi", # Kỳ vọng: POSITIVE
        "that su that vong", # Kỳ vọng: NEGATIVE
        "cung duoc, khong den noi te", # Kỳ vọng: NEUTRAL hoặc POSITIVE nhẹ (do "khong_te")
        "giao hang nhanh, dong goi ky", # Kỳ vọng: POSITIVE
        "dien thoai chay on dinh", # Kỳ vọng: POSITIVE
        "khong hai long ve san pham", # Kỳ vọng: NEGATIVE
        "mau sac san pham khong giong hinh" # Kỳ vọng: NEGATIVE hoặc NEUTRAL
    ]

    for comment in test_comments:
        prediction = classifier.predict(comment)
        print(f"Comment: '{comment}' => Predicted Sentiment: {prediction}")
