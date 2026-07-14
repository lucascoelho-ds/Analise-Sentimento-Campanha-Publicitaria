# nlp-u3a1-aplicacoes

# Análise Básica de Sentimentos com spaCy e Naive Bayes

Script em Python que treina um classificador de sentimentos (positivo, negativo, neutro) a partir de comentários em português, usando pré-processamento linguístico com spaCy e um modelo Naive Bayes (scikit-learn).

# Como funciona


1. Carregamento do modelo de linguagem
Usa o modelo pt_core_news_sm do spaCy para processar texto em português (tokenização, lematização, identificação de stopwords e pontuação).

2. Geração da base de dados
Um pequeno conjunto de frases-exemplo é definido para cada categoria (positivo, negativo, neutro). O script sorteia aleatoriamente 500 frases desse conjunto para montar um dataset rotulado (comentario, sentimento).

 ⚠️ Como as frases são sorteadas com repetição a partir de um conjunto muito pequeno (5 frases por categoria), o dataset final contém muitas frases duplicadas. Isso é o que explica a acurácia de 100% no relatório — o modelo está memorizando frases já vistas, não generalizando para linguagem nova. Veja a seção Limitações abaixo.

3. Pré-processamento (limpar_texto)
Cada comentário é convertido para minúsculas e processado pelo spaCy, que:

 - reduz cada palavra ao seu lema (forma base);
 - remove pontuação, espaços e stopwords (palavras muito comuns e pouco informativas, como "de", "que", "o").

4. Vetorização (Bag of Words)
O CountVectorizer do scikit-learn transforma os textos limpos em vetores numéricos de contagem de palavras.

5. Treinamento
Os dados são divididos em treino (75%) e teste (25%) com train_test_split, e um classificador MultinomialNB (Naive Bayes multinomial, adequado para contagens de palavras) é treinado sobre os dados de treino.

6. Predição em novos comentários
Três comentários novos, escritos manualmente, são limpos, vetorizados e classificados pelo modelo treinado.

7. Avaliação
O script imprime:
 - as predições para os novos comentários;
 - a acurácia do modelo no conjunto de teste;
 - um relatório de classificação completo (precision, recall, f1-score por classe).

# Requisitos
- Python 3.8+
- Bibliotecas:

+ bash:  pip install pandas spacy scikit-learn

- Modelo de português do spaCy:

+ bash:  python -m spacy download pt_core_news_sm

# Como executar

+ bash: python analise_sentimentos.py

# Exemplo de saída

--- RESULTADOS DA ANÁLISE BÁSICA DE SENTIMENTOS ---

Comentário: Estou amando as novidades! | Sentimento Predito: positivo
Comentário: Que porcaria de serviço. | Sentimento Predito: negativo
Comentário: Achei mediano, nada de especial. | Sentimento Predito: neutro

Acurácia do modelo nos dados de teste: 100.00%

--- MÉTRICAS DE DESEMPENHO PROFISSIONAIS ---

Relatório de Classificação:

              precision    recall  f1-score   support

    negativo       1.00      1.00      1.00        41
      neutro       1.00      1.00      1.00        40
    positivo       1.00      1.00      1.00        44

    accuracy                           1.00       125
   macro avg       1.00      1.00      1.00       125
weighted avg       1.00      1.00      1.00       125

Acurácia nos dados de teste: 100.00%

# Limitações importantes

 - Dataset artificial e repetitivo: as 500 "amostras" vêm de apenas 5 frases fixas por categoria, sorteadas com repetição. Na prática, treino e teste contêm as mesmas frases repetidas várias vezes, o que infla artificialmente a acurácia para 100%. Isso não reflete a capacidade real do modelo de generalizar para comentários nunca vistos.
 - Vocabulário limitado: como o vetorizador só "aprende" as palavras presentes nessas 15 frases-base, qualquer comentário novo com vocabulário muito diferente tende a ser mal classificado.
 - Objetivo didático: este código serve como demonstração do fluxo completo de um pipeline de NLP (pré-processamento → vetorização → treino → avaliação), não como um classificador de sentimentos pronto para produção.


# Possíveis melhorias

 - Usar uma base de dados real e maior (ex.: reviews rotulados, datasets públicos de sentimento em português).
 - Separar frases-base diferentes para treino e "novos comentários" de teste, garantindo que o teste realmente avalie generalização.
 - Testar vetorizadores mais robustos, como TfidfVectorizer.
 - Testar outros modelos (Regressão Logística, SVM) e comparar desempenho.
 - Adicionar validação cruzada (cross_val_score) para uma estimativa mais confiável da performance.


# Estrutura do fluxo (resumo)

Comentários brutos
      │
      ▼
Limpeza com spaCy (lemmatização, remoção de stopwords/pontuação)
      │
      ▼
Vetorização (CountVectorizer)
      │
      ▼
Treino do modelo (MultinomialNB)
      │
      ▼
Predição e avaliação (acurácia, classification_report)