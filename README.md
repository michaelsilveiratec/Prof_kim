Sistema Preditivo de Palavras

Exercício-Programa da disciplina Análise Exploratória de Dados (5º semestre). O sistema lê uma frase e sugere a palavra seguinte mais provável, informando a confiança da predição.

A abordagem não é uma rede neural. É um modelo de linguagem clássico: frequências de n-grams, Teorema de Bayes e suavização de Laplace. A ideia é tornar explícito o que o modelo está calculando e por que uma palavra ganha de outra.

$ python -m src.main "O aluno estudou para a prova de"
Sugestão: Estatística, Confiança: 59.27%

A confiança é a probabilidade posterior da palavra escolhida, considerando também as hipóteses de continuação desconhecida e fim de sentença. Assim, o smoothing não atribui probabilidade zero a uma continuação inédita.

Para ver as três melhores hipóteses:

$ python -m src.main "O aluno estudou para a prova de" --top 3
Sugestão: Estatística, Confiança: 59.27%
Sugestão: Probabilidade, Confiança: 8.09%
Sugestão: Cálculo, Confiança: 5.91%

## Modelagem probabilística

O modelo extrai frequências de unigramas, bigramas e trigramas por sentença. Para
um contexto conhecido `C`, suaviza a probabilidade de cada continuação:

`P(W | C) = (count(C, W) + alpha) / (count(C) + alpha * K)`

`K` é a quantidade de continuações observadas para aquele contexto, mais uma
categoria agregada `UNK` para palavras inéditas e, quando aplicável, o evento
de fim de sentença. O parâmetro `alpha` é positivo e configurável pela opção
`--alpha` (padrão `1.0`).

A implementação expressa essa estimativa pela regra de Bayes. Usa o prior
smoothed da palavra `P(W)` e estima `P(C)` pela frequência do contexto; então
obtém `P(C | W) = P(W | C) * P(C) / P(W)` e calcula
`P(W | C) = P(C | W) * P(W) / P(C)`, normalizando pela evidência sobre as
hipóteses de continuação. Essa forma combina a contagem de n-grams com o
Teorema de Bayes sem estimar diretamente uma probabilidade reversa instável
num corpus pequeno.

Quando os contextos completo e menor estão disponíveis, suas distribuições são
interpoladas (80% para a ordem maior e 20% para a menor). Se o contexto completo
não foi observado, o preditor usa o sufixo conhecido mais longo. Se nenhum
contexto correspondente existe, usa o prior smoothed do corpus. Entrada com
palavras fora do vocabulário não causa erro: o preditor recua para sufixos
conhecidos e, se não houver nenhum, usa o prior. A categoria `UNK` preserva
probabilidade para continuações não vistas.

------------------------------------------------------------------------------------------------

***Como rodar***
Requisito: Python 3.10 ou superior. Não há dependências externas; o núcleo usa só a biblioteca padrão.

No terminal digite:
git clone https://github.com/michaelsilveiratec/Prof_kim.git

em seguida:
cd AP2---AED-Parte-1

python -m src.main "O aluno estudou para a prova de"

python -m src.main "A análise exploratória de" --top 3

python -m src.main --interactive

python -m src.main --evaluate

python -m unittest discover -s tests

O relatório atual do corpus indica top-1 de 100% nos 10 casos rotulados e
top-1 de 41,4% / top-3 de 48,3% no hold-out de 29 sentenças. Os casos rotulados
medem as frases-alvo do projeto; o hold-out oferece uma estimativa mais realista
e evidencia que a qualidade depende do tamanho e da variedade do corpus.

------------------------------------------------------------------------------------------------
Flag                  Efeito

frase                 Texto cujo próximo token será predito
-i, --interactive     Loop no terminal; digite sair para encerrar
-e, --evaluate        Relatório de precisão nos casos rotulados e no hold-out
-k, --top             Quantidade de sugestões no ranking
-n, --order           Ordem máxima do n-gram (padrão: 3)
-a, --alpha           Parâmetro da suavização de Laplace (padrão: 1.0)
-c, --corpus          Caminho de um corpus próprio

------------------------------------------------------------------------------------------------

Organização do repositório

src/
  __init__.py        inicialização do pacote
  __main__.py        ponto de entrada secundário para execução via módulo
  config.py          configurações padrão e caminhos do sistema
  preprocessing.py   limpeza, normalização e tokenização
  ngrams.py          extrator de n-grams, contagens de frequências e vocabulário
  bayes.py           preditor bayesiano (prior, likelihood, evidência e smoothing)
  evaluator.py       avaliador de precisão nos casos de teste e hold-out
  main.py            interface principal de linha de comando (CLI)
data/
  corpus.txt         corpus textual para treinamento do modelo
tests/
  __init__.py        inicialização do pacote de testes
  test_preprocessing.py testes de unidade do pré-processamento
  test_ngrams.py     testes de unidade da extração de n-grams
  test_bayes.py      testes de unidade do modelo bayesiano
  test_integration.py testes de integração do fluxo completo
.gitignore           arquivos e diretórios ignorados pelo Git
requirements.txt     arquivo de dependências do projeto
README.md            documentação técnica e guia do repositório