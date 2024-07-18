import joblib

class TopicClassifier(object):
    def __init__(self):
        self.vectorizer = joblib.load('text_vectorizer.pkl')
        self.model = joblib.load('text_classification_model.pkl')
        self.labels = [
             'alt.atheism',
             'comp.graphics',
             'comp.os.ms-windows.misc',
             'comp.sys.ibm.pc.hardware',
             'comp.sys.mac.hardware',
             'comp.windows.x',
             'misc.forsale',
             'rec.autos',
             'rec.motorcycles',
             'rec.sport.baseball',
             'rec.sport.hockey',
             'sci.crypt',
             'sci.electronics',
             'sci.med',
             'sci.space',
             'soc.religion.christian',
             'talk.politics.guns',
             'talk.politics.mideast',
             'talk.politics.misc',
             'talk.religion.misc']

    def predict(self, text):
        vector = self.vectorizer.transform([text])
        label = self.model.predict(vector)[0]
        return self.labels[label]



















    