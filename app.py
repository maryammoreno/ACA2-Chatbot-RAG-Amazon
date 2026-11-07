
import streamlit as st
import chromadb
from google import genai
import fitz
import os

# Configuración de la aplicación
st.set_page_config(
    page_title="Chatbot RAG - Amazon España",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chatbot RAG - Amazon España")
st.write("Consulta las condiciones de uso y venta de Amazon España.")

# Configuración segura de Gemini
@st.cache_resource
def configurar_gemini():
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

gemini_client = configurar_gemini()

# Configuración de ChromaDB
@st.cache_resource
def configurar_chromadb():
    chroma_client = chromadb.Client()
    coleccion = chroma_client.get_or_create_collection(
        name="condiciones_amazon",
        metadata={"hnsw:space": "cosine"}
    )
    return coleccion

collection = configurar_chromadb()

# Lectura y fragmentación del documento PDF
@st.cache_data
def cargar_documento():
    ruta_pdf = "condiciones_amazon.pdf"

    documento = fitz.open(ruta_pdf)

    texto = ""
    for pagina in documento:
        texto += pagina.get_text()

    documento.close()

    # Fragmentación utilizada en el ACA 1
    tamaño_fragmento = 1000
    superposicion = 150

    fragmentos = []

    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamaño_fragmento
        fragmentos.append(texto[inicio:fin])
        inicio += tamaño_fragmento - superposicion

    return fragmentos


# Almacenar los fragmentos en ChromaDB
def preparar_base_vectorial():

    fragmentos = cargar_documento()

    if collection.count() == 0:

        for inicio in range(0, len(fragmentos), 20):

            lote = fragmentos[inicio:inicio + 20]

            collection.upsert(
                ids=[
                    f"fragmento_{i}"
                    for i in range(inicio, inicio + len(lote))
                ],
                documents=lote,
                metadatas=[
                    {
                        "documento": "Condiciones de Uso y Venta de Amazon",
                        "fecha": "2025-08-08",
                        "fragmento": i
                    }
                    for i in range(inicio, inicio + len(lote))
                ]
            )

    return collection.count()


total_fragmentos = preparar_base_vectorial()

st.sidebar.success(
    f"Documento cargado: {total_fragmentos} fragmentos"
)

# Modelos candidatos para Gemini
MODELOS_CANDIDATOS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]


# Generar respuestas con selección automática del modelo
def generar_con_gemini(prompt):

    from google.genai import errors

    for modelo in MODELOS_CANDIDATOS:
        try:
            respuesta = gemini_client.models.generate_content(
                model=modelo,
                contents=prompt
            )

            if respuesta.text:
                return respuesta.text

        except errors.ClientError as e:
            if e.code in [404, 429]:
                continue
            raise

    raise RuntimeError(
        "No hay modelos de Gemini disponibles para esta consulta."
    )


# Consultar el sistema RAG
def consultar_rag(pregunta):

    resultados = collection.query(
        query_texts=[pregunta],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    contexto = ""
    fuentes = []

    for i in range(len(resultados["ids"][0])):

        identificador = resultados["ids"][0][i]
        fragmento = resultados["documents"][0][i]
        metadatos = resultados["metadatas"][0][i]

        contexto += f"""
ID: {identificador}
Documento: {metadatos.get('documento', 'No especificado')}
Fecha: {metadatos.get('fecha', 'No especificada')}
Contenido:
{fragmento}

"""

        fuentes.append({
            "id": identificador,
            "contenido": fragmento,
            "metadatos": metadatos,
            "similitud": round(
                (1 - resultados["distances"][0][i]) * 100, 2
            )
        })

    prompt_rag = f"""
Eres un asistente especializado en las condiciones
de uso y venta de Amazon España.

Responde utilizando únicamente la información
proporcionada en el contexto documental.

Si la información es insuficiente, responde:
"No se encontró información suficiente en el
documento consultado para responder esta pregunta."

No inventes información ni utilices conocimientos externos.

Cita las fuentes utilizando sus identificadores reales,
por ejemplo: [fragmento_52].

No cites fragmentos que no respalden la respuesta.

CONTEXTO DOCUMENTAL:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:
"""

    respuesta = generar_con_gemini(prompt_rag)

    return respuesta, fuentes

# Historial de conversación
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar los mensajes anteriores
for mensaje in st.session_state.mensajes:

    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

        if mensaje["rol"] == "assistant":
            with st.expander("📚 Consultar fuentes"):
                for fuente in mensaje["fuentes"]:
                    st.write(f"**ID:** {fuente['id']}")
                    st.write(f"**Similitud:** {fuente['similitud']} %")
                    st.write(f"**Metadatos:** {fuente['metadatos']}")
                    st.write(fuente["contenido"])
                    st.divider()


# Campo para escribir preguntas
pregunta = st.chat_input(
    "Escribe tu pregunta sobre las condiciones de Amazon..."
)

if pregunta:

    # Guardar y mostrar la pregunta
    st.session_state.mensajes.append({
        "rol": "user",
        "contenido": pregunta
    })

    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar respuesta
    with st.chat_message("assistant"):

        with st.spinner("Consultando el documento..."):
            try:
                respuesta, fuentes = consultar_rag(pregunta)

                st.markdown(respuesta)

                with st.expander("📚 Consultar fuentes"):
                    for fuente in fuentes:
                        st.write(f"**ID:** {fuente['id']}")
                        st.write(f"**Similitud:** {fuente['similitud']} %")
                        st.write(f"**Metadatos:** {fuente['metadatos']}")
                        st.write(fuente["contenido"])
                        st.divider()

                # Guardar respuesta en el historial
                st.session_state.mensajes.append({
                    "rol": "assistant",
                    "contenido": respuesta,
                    "fuentes": fuentes
                })

            except Exception as e:
                st.error(
                    "No fue posible generar la respuesta. "
                    "Verifica la configuración de Gemini "
                    "o la disponibilidad del modelo."
                )

# Exploración del documento
st.sidebar.divider()
st.sidebar.subheader("📄 Documento disponible")

st.sidebar.write(
    "Condiciones de Uso y Venta de Amazon España"
)
st.sidebar.caption("Fecha del documento: 08/08/2025")

# Permitir descargar el documento original
with open("condiciones_amazon.pdf", "rb") as archivo:
    st.sidebar.download_button(
        label="📥 Descargar documento PDF",
        data=archivo.read(),
        file_name="condiciones_amazon.pdf",
        mime="application/pdf"
    )

# Explorar los fragmentos almacenados
with st.sidebar.expander("🔎 Explorar fragmentos"):

    ids = collection.get()["ids"]

    fragmento_elegido = st.selectbox(
        "Selecciona un fragmento:",
        sorted(ids, key=lambda x: int(x.split("_")[1]))
    )

    if fragmento_elegido:
        resultado = collection.get(
            ids=[fragmento_elegido],
            include=["documents", "metadatas"]
        )

        st.write("**Metadatos:**")
        st.json(resultado["metadatas"][0])

        st.write("**Contenido:**")
        st.write(resultado["documents"][0])
        
