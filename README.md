# Analizador de Portafolio

Aplicación Streamlit para analizar movimientos de un portafolio y compararlo con el ICOLCAP.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Archivo de entrada

Carga un CSV que incluya estas columnas:

- `Fecha y hora` — por ejemplo: `2 sept 2026, 9:10 a. m.`
- `Tipo de movimiento` — `Depósito` o `Retiro`
- `Estado` — se incluyen únicamente los registros `Aprobado`
- `Valor total`

La aplicación solicita el valor actual del portafolio antes de calcular el XIRR y la rentabilidad.
