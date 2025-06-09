import re

def extract_features(email_text):
    #  Count num_links using regex
    num_links = len(re.findall(r'http[s]?://\S+', email_text))

    #  Count num_words
    words = email_text.split()
    num_words =len(words)

    #  Check for promotion keyboards
    promo_keywords = ['offer', 'buy now', 'free', 'discount', 'limited time']
    has_offer = int(any(keyword in email_text.lower() for keyword in promo_keywords))

    # Count words in all_caps
    all_cap_words = [word for word in words if word.isupper() and len(word) > 1]
    all_cap = int(len(all_cap_words) >= 3)

    # Return features list
    return[[num_links, num_words, has_offer, all_cap]]


email_text = input('Enter email text: ')
print(extract_features(email_text))
