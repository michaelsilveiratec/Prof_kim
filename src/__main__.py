import os
from src.preprocessing import TextPreprocessor
from src.ngrams import NGramModel
from src.bayes import BayesianPredictor

def main():
    corpus_path = os.path.join("data", "corpus.txt")
    
    # Corpus de contingência/exemplo caso o arquivo não exista
    default_text = (
        "o aluno estudou para a prova de estatistica "
        "o aluno estudou para a prova de matematica "
        "o aluno tirou nota boa na prova de estatistica"
    )
    
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
    else:
        corpus_text = default_text

    # 1. Pré-processamento
    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(corpus_text)

    # 2. Treinamento de N-Grams
    model = NGramModel(n=3)
    model.fit(tokens)

    # 3. Predição Bayesiana
    predictor = BayesianPredictor(model, alpha=1.0)
    
    # Exemplo de entrada exigido no enunciado
    frase_input = "O aluno estudou para a prova de"
    input_tokens = preprocessor.tokenize(frase_input)
    
    palavra, probabilidade = predictor.predict_next_word(input_tokens)

    # Exibição conforme especificado no Requisito 4
    print(f"Sugestão: {palavra.capitalize()}, Confiança: {probabilidade:.2f}%")

if __name__ == "__main__":
    main()