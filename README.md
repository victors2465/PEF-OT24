# PEF-OT24
Este repositorio contiene los archivos realizdos durante el periodo OT-24, en el que estan los archivos para la extracción de la región de interes (ROI) asi como tambien los archivos de machine learning desarrollados.

---

### Estructura del repositorio

1. Carpetas de modelos de clasificación.
    - **D_HC_MCI_modelos:**  Carpeta con los archivos para la clasificación entre Demencia, Healthy Control (HC) y Mild Cognitive Impairement (MCI).
    Se incluyen los archivos de: 

        - bagging.ipynb
        - boosting.ipynb
        - full_code_final.ipynb
        - informacion_completa_imagenes_V2.csv

        Los archivos de full_code_final.ipynb y el archivo de datos csv se repiten a lo largo de los 4 repositorios, dentre del archivo ipynb se encuentran los modelos de bagging, catboost, LGBM y sus respectivas métricas 
    - **D_MCI_pacientes_parkinson_Modelos:**  Carpeta con los archivos para la clasificación entre Demencia (D) y Mild Cognitive Impairement (MCI).
    - **DMCI_v_NC_modelos:**  Carpeta con los archivos para la clasificación entre Demencia (D) y Mild Cognitive Impairement (MCI) de manera conjunta contra Normal Condition (NC).
    - **parkinson_modelos:** Carpeta con los archivos para la clasificación entre personas con Parkinson y personas sin Parkinson.

2. Carpeta PEF-PR24.
    Contiene los archivos desarrollados priniciaples a lo largo del periodo mencionado, estos archivos fueron modificados a lo largo de OT-24, la funciolaidad de estos archvivos es realizar la separacion en componentes de los relojes y analizarlos, asi como el modelo entrenado de YOLO.  

3. Archivos individuales.
    - **ROI_Manual.py:** Su funcionalidad es realizar la extraccion de la región de interes de manera manual seleccionando cuantro puntos en pantalla.
    - **ROI. py :** su funcionalidad es relizar la extraccion automatica de la región de interes.
    - **PDF_2_PNG.py:** Su función es convertir los archivos PDF a PNG
    - **requirements.txt:** Recursos necesarios para la ejecución correcta de los archivos mencionados.