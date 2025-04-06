# 📦 BBS - Otimização de Armazenamento de Documentos

Este projeto realiza análises estatísticas sobre movimentações documentais de empresas parceiras, e gera recomendações para reorganização de caixas em estantes/prateleiras, com foco em eficiência e economia de tempo.

## 🔧 Funcionalidades

- Carregamento de dados a partir de planilhas Excel.
- Cálculo de frequência de requisições por empresa.
- Regressão linear para prever economia de tempo.
- Sugestões automáticas de estantes e prateleiras ideais.
- Exportação completa dos resultados em um novo arquivo Excel.
- Interface gráfica amigável com barra de progresso.

## 📁 Estrutura Esperada do Excel de Entrada

- **Aba "Empresas parceiras"**: lista de empresas.
- **Aba "Movimentação"**: registros de requisição e movimentação de documentos.
- **Aba "Tipos de movimentações"**: classificação das movimentações.

## 🚀 Como usar

### 1. Com Python instalado:

```bash
pip install -r requirements.txt
python main.py
```

### 2. Sem Python (usuários finais):
Baixe o executável .exe na na aba [Releases](https://github.com/Ruben-BG/StorageOptimization/releases)
