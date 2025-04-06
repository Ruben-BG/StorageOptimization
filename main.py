import os
import numpy as np
import pandas as pd
import FreeSimpleGUI as fg
from scipy import stats
from sklearn.linear_model import LinearRegression

class DocumentManagementAnalyzer:
    def __init__(self):
        self.df_movimentacao = None
        self.df_empresas = None
        self.df_tipos = None
        self.company_analysis = None
        self.regression_model = None
        self.regression_results = None
    
    def load_data(self, excel_path):
        """Carrega os dados das planilhas Excel"""
        try:
            self.df_empresas = pd.read_excel(excel_path, sheet_name='Empresas parceiras')
            self.df_movimentacao = pd.read_excel(excel_path, sheet_name='Movimentação')
            self.df_tipos = pd.read_excel(excel_path, sheet_name='Tipos de movimentações')
            
            self.df_movimentacao['Data do requerimento'] = pd.to_datetime(self.df_movimentacao['Data do requerimento'])
            self.df_movimentacao['Data da conclusão do requerimento'] = pd.to_datetime(self.df_movimentacao['Data da conclusão do requerimento'])
            
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
        
    def analyze_frequency(self):
        try:
            freq_company = self.df_movimentacao["Empresa"].value_counts().reset_index()
            freq_company.columns = ["Empresa", "Frequência"]
            freq_company["Empresa"] = freq_company["Empresa"].astype(str)
            
            racks_by_company = self.df_movimentacao.groupby("Empresa")["Estante alterada"].mean().reset_index()
            racks_by_company.columns = ["Empresa", "Estante média"]
            racks_by_company["Empresa"] = racks_by_company["Empresa"].astype(str)
            
            shelves_by_company = self.df_movimentacao.groupby("Empresa")["Prateleira alterada"].mean().reset_index()
            shelves_by_company.columns = ["Empresa", "Prateleira média"]
            shelves_by_company["Empresa"] = shelves_by_company["Empresa"].astype(str)
            
            self.df_movimentacao['Tempo atendimento'] = (
                self.df_movimentacao['Data da conclusão do requerimento'] - 
                self.df_movimentacao['Data do requerimento']
            ).dt.total_seconds() / 3600
            
            time_by_company = self.df_movimentacao.groupby("Empresa")["Tempo atendimento"].mean().reset_index()
            time_by_company.columns = ["Empresa", "Tempo médio atendimento (h)"]
            
            def calculate_position_score(row):
                prateleira_weight = 13 - row['Prateleira alterada']
                estante_weight = row['Estante alterada']
                return (prateleira_weight * 0.7) + (estante_weight * 0.3)
            
            self.df_movimentacao['Posição Score'] = self.df_movimentacao.apply(calculate_position_score, axis=1)
            position_by_company = self.df_movimentacao.groupby("Empresa")["Posição Score"].mean().reset_index()
            position_by_company.columns = ["Empresa", "Dificuldade Posição Atual"]
            
            freq_company["Empresa"] = freq_company["Empresa"].astype(str)
            time_by_company["Empresa"] = time_by_company["Empresa"].astype(str)
            position_by_company["Empresa"] = position_by_company["Empresa"].astype(str)

            company_analysis = freq_company.merge(time_by_company, on="Empresa").merge(position_by_company, on="Empresa")
            
            company_analysis = company_analysis.merge(
                self.df_empresas[['Nome da empresa', 'Identificador']], 
                left_on='Empresa', 
                right_on='Nome da empresa',
                how='left'
            ).drop(columns=['Nome da empresa'])
            
            self.company_analysis = company_analysis
            return True
        except Exception as e:
            print(f"Erro ao analisar frequência: {e}")
            return False
    
    def analyze_requesting_company(self):
        try:
            total_requests = self.company_analysis["Frequência"].sum()
            self.company_analysis["Probabilidade"] = self.company_analysis["Frequência"] / total_requests
            
            if len(self.company_analysis["Frequência"]) >= 8:
                _, p_value = stats.normaltest(self.company_analysis["Frequência"])
            else:
                p_value = np.nan

            self.company_analysis["Distribuição Normal (p-value)"] = p_value
            
            self.company_analysis = self.company_analysis.sort_values(by="Frequência", ascending=False)
            
            def calculate_recommendation(row, max_estante=48, max_prateleira=12):
                freq_norm = (row['Frequência'] - self.company_analysis['Frequência'].min()) / \
                           (self.company_analysis['Frequência'].max() - self.company_analysis['Frequência'].min())
                
                estante_rec = max(1, round(max_estante * (1 - freq_norm * 0.8)))
                
                prateleira_rec = max(1, round(max_prateleira * (1 - freq_norm)))
                
                return pd.Series([estante_rec, prateleira_rec])
            
            self.company_analysis[['Estante recomendada', 'Prateleira recomendada']] = \
                self.company_analysis.apply(calculate_recommendation, axis=1)
            
            X = self.company_analysis[['Frequência', 'Dificuldade Posição Atual']]
            y = self.company_analysis['Tempo médio atendimento (h)']
            
            self.regression_model = LinearRegression()
            self.regression_model.fit(X, y)
            
            self.regression_results = {
                'R²': self.regression_model.score(X, y),
                'Coeficientes': self.regression_model.coef_,
                'Intercept': self.regression_model.intercept_
            }
            
            self.company_analysis['Dificuldade Posição Atual'] = \
                (13 - self.company_analysis['Prateleira recomendada']) * 0.7 + \
                self.company_analysis['Estante recomendada'] * 0.3
                
            X_new = self.company_analysis[['Frequência', 'Dificuldade Posição Atual']]
            self.company_analysis['Tempo previsto com recomendação (h)'] = self.regression_model.predict(X_new)
            
            self.company_analysis['Economia de tempo estimada (h)'] = \
                self.company_analysis['Tempo médio atendimento (h)'] - \
                self.company_analysis['Tempo previsto com recomendação (h)']
            
            return True
        except Exception as e:
            print(f"Erro ao analisar probabilidade de empresas: {e}")
            return False
    
    def analyze_movement_types(self):
        """Analisa os tipos de movimentação mais frequentes e seu impacto"""
        try:
            self.df_tipos = self.df_tipos.rename(columns={'Tipos de movimentações': 'Tipo de movimentação'})
            self.df_movimentacao["Tipo de movimentação"] = self.df_movimentacao["Tipo de movimentação"].astype(str)
            self.df_tipos["Tipo de movimentação"] = self.df_tipos["Tipo de movimentação"].astype(str)
            movimentacao_com_tipos = self.df_movimentacao.merge(
                self.df_tipos,
                left_on='Tipo de movimentação',
                right_on='Tipo de movimentação',
                how='left'
            )
            
            freq_tipos = movimentacao_com_tipos['Tipo de movimentação'].value_counts().reset_index()
            freq_tipos.columns = ['Tipo de movimentação', 'Frequência']
            
            tempo_tipos = movimentacao_com_tipos.groupby('Tipo de movimentação')['Tempo atendimento'].mean().reset_index()
            tempo_tipos.columns = ['Tipo de movimentação', 'Tempo médio (h)']
            
            analysis_tipos = freq_tipos.merge(tempo_tipos, on='Tipo de movimentação')
            
            return analysis_tipos
        except Exception as e:
            print(f"Erro ao analisar tipos de movimentação: {e}")
            return None
    
    def export_results(self, output_path):
        """Exporta os resultados para um arquivo Excel"""
        try:
            with pd.ExcelWriter(output_path) as writer:
                self.company_analysis.to_excel(writer, sheet_name='Análise Empresas', index=False)
                
                tipos_analysis = self.analyze_movement_types()
                if tipos_analysis is not None:
                    tipos_analysis.to_excel(writer, sheet_name='Análise Tipos Movimentação', index=False)
                
                regression_df = pd.DataFrame({
                    'Métrica': ['R²', 'Coeficiente Frequência', 'Coeficiente Dificuldade', 'Intercepto'],
                    'Valor': [
                        self.regression_results['R²'],
                        self.regression_results['Coeficientes'][0],
                        self.regression_results['Coeficientes'][1],
                        self.regression_results['Intercept']
                    ]
                })
                regression_df.to_excel(writer, sheet_name='Regressão Linear', index=False)
                
                summary_data = {
                    'Total de Empresas Analisadas': [len(self.company_analysis)],
                    'Empresa Mais Frequente': [self.company_analysis.iloc[0]['Empresa']],
                    'Frequência da Empresa Mais Ativa': [self.company_analysis.iloc[0]['Frequência']],
                    'Economia Total de Tempo Estimada (h)': [self.company_analysis['Economia de tempo estimada (h)'].sum()],
                    'Tipos de Movimentação Mais Comuns': [tipos_analysis.iloc[0]['Tipo de movimentação'] if tipos_analysis is not None else 'N/A']
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumo Executivo', index=False)
            
            return True
        except Exception as e:
            print(f"Erro ao exportar resultados: {e}")
            return False

def create_gui():
    """Cria a interface gráfica do aplicativo"""
    fg.theme('LightGrey1')
    
    layout = [
        [fg.Text('Sistema de Análise para Gestão Documental', font=('Helvetica', 16))],
        [fg.HorizontalSeparator()],
        [
            fg.Text('Arquivo Excel:'), 
            fg.Input(key='-FILE-'), 
            fg.FileBrowse(file_types=(("Excel Files", "*.xlsx"),))
        ],
        [fg.Button('Carregar Dados'), fg.Button('Analisar Dados', disabled=True), fg.Button('Exportar Resultados', disabled=True)],
        [fg.Multiline(size=(80, 20), key='-OUTPUT-', autoscroll=True, disabled=True)],
        [fg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-')],
        [fg.Button('Sair')]
    ]
    
    window = fg.Window('BBS - Otimização de Armazenamento', layout, finalize=True)
    
    analyzer = DocumentManagementAnalyzer()
    analysis_done = False
    
    while True:
        event, values = window.read()
        
        if event == fg.WINDOW_CLOSED or event == 'Sair':
            break
        
        elif event == 'Carregar Dados':
            if values['-FILE-'] and os.path.exists(values['-FILE-']):
                window['-OUTPUT-'].update("Carregando dados...\n")
                window['-PROGRESS-'].update(30)
                
                success = analyzer.load_data(values['-FILE-'])
                
                if success:
                    window['-OUTPUT-'].print("Dados carregados com sucesso!\n")
                    window['-PROGRESS-'].update(100)
                    window['Analisar Dados'].update(disabled=False)
                else:
                    window['-OUTPUT-'].print("Erro ao carregar dados. Verifique o arquivo.\n")
                    window['-PROGRESS-'].update(0)
            else:
                window['-OUTPUT-'].print("Por favor, selecione um arquivo válido.\n")
        
        elif event == 'Analisar Dados':
            window['-OUTPUT-'].print("Iniciando análise estatística...")
            window['-PROGRESS-'].update(10)
            
            frequency_result = analyzer.analyze_frequency()
            
            if frequency_result:
                window['-PROGRESS-'].update(40)
                window['-OUTPUT-'].print("Análise de frequência concluída...")
                
                company_analyse_result = analyzer.analyze_requesting_company()
                
                if company_analyse_result:
                    window['-PROGRESS-'].update(80)
                    window['-OUTPUT-'].print("Análise de probabilidade e regressão concluída...")
                    
                    top_company = analyzer.company_analysis.iloc[0]
                    window['-OUTPUT-'].print(f"\nEmpresa mais ativa: {top_company['Empresa']}")
                    window['-OUTPUT-'].print(f"Frequência: {top_company['Frequência']} pedidos")
                    window['-OUTPUT-'].print(f"Probabilidade: {top_company['Probabilidade']*100:.2f}%")
                    window['-OUTPUT-'].print(f"Recomendação: Estante {top_company['Estante recomendada']}, Prateleira {top_company['Prateleira recomendada']}")
                    window['-OUTPUT-'].print(f"Economia de tempo estimada: {top_company['Economia de tempo estimada (h)']:.2f} horas\n")
                    
                    analysis_done = True
                    window['Exportar Resultados'].update(disabled=False)
                    window['-PROGRESS-'].update(100)
                else:
                    window['-OUTPUT-'].print("Erro na análise de probabilidade. Verifique os dados da planilha 'Movimentação'.\n")
                    window['-PROGRESS-'].update(0)
            else:
                window['-OUTPUT-'].print("Erro na análise de frequência. Verifique os dados da planilha 'Movimentação'.\n")
                window['-PROGRESS-'].update(0)
        
        elif event == 'Exportar Resultados' and analysis_done:
            output_file = fg.popup_get_file(
                'Salvar como', save_as=True, 
                file_types=(("Excel Files", "*.xlsx"),),
                default_extension=".xlsx")
            
            if output_file:
                window['-OUTPUT-'].print("Exportando resultados...\n")
                window['-PROGRESS-'].update(50)
                
                success = analyzer.export_results(output_file)
                
                if success:
                    window['-OUTPUT-'].print(f"Resultados exportados para: {output_file}\n")
                    window['-PROGRESS-'].update(100)
                else:
                    window['-OUTPUT-'].print("Erro ao exportar resultados.\n")
                    window['-PROGRESS-'].update(0)
    
    window.close()

if __name__ == '__main__':
    create_gui()