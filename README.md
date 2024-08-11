# **Mini-Projeto** - Reconhecimento de Dígitos
### Aluno: Thiago Rodrigues Cruz Justino
### Professores: Giberto Farias e Bruno Jefferson de Sousa

Inicialmente, é realizado o tratamento dos arquivos encontrados na pasta dataset_digits, que contém train_csv e test_csv. Esses arquivos possuem a primeira coluna como o rótulo (0, 1, 4 ou 5) e as 784 colunas restantes representando os valores dos pixels, que variam de 0 a 255. Para simplificar o problema, que originalmente possui 784 dimensões, são criadas duas novas colunas: simetria (a soma da simetria horizontal e vertical) e intensidade. Essas novas características são usadas no lugar dos valores dos pixels. Embora essa abordagem resulte em uma perda de informação, ela reduz a dimensionalidade do problema e o torna mais visualizável graficamente, uma vez que agora são utilizadas apenas duas "features". Abaixo está uma imagem do gráfico que mostra as labels e seus respectivos pontos utilizando intensidade e simetria.
![Gráfico](imagens/grafico_int_simetria.png)
