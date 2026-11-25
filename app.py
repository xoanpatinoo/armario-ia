import io
import json
import os
import re
from pathlib import Path
from typing import Literal

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from supabase import Client, create_client

# ==============================================================================
# 1. CONFIGURACION DE RUTAS Y ENTORNO (ESCRITORIO VS NUBE/MOVIL)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
CARPETA_ARMARIO = BASE_DIR / "mi_armario"
CARPETA_REFERENCIAS = BASE_DIR / "referencias_estilo"
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}
BUCKET_NAME = "vestidor"
MODELO_GEMINI = "gemini-2.5-flash"

EN_NUBE = "/mount/src" in str(BASE_DIR).replace("\\", "/")

if not EN_NUBE:
    CARPETA_ARMARIO.mkdir(parents=True, exist_ok=True)
    CARPETA_REFERENCIAS.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CEREBRO DEL ESTILISTA (SYSTEM PROMPT CON PERFIL Y REGLAS EXACTAS)
# ==============================================================================
SYSTEM_PROMPT_ESTILISTA = """
Eres un Estilista Masculino Senior experto en colorimetria personal, proporciones corporales y sastreria casual contemporanea.
Tu unico cliente tiene estas caracteristicas fisicas, geograficas y de estilo INNEGOCIABLES:

1. FISICO Y ENTORNO:
- Hombre de 22 anos, 1,83 m de altura, 85 kg, complexion ancha y atletica (hombros y torso desarrollados).
- Reside entre Vilagarcia de Arousa y Santiago de Compostela (clima humedo, lluvioso y fresco en otono/invierno).

2. COLORIMETRIA: OTONO CALIDO (TRUE AUTUMN)
- Piel con subtono calido, ojos marrones, cabello castano claro con reflejos dorados/cobrizos naturales y barba con matices calidos.
- COLORES ESTRELLA (maxima armonia): Marron chocolate, camel, arena/beige calido, verde oliva/musgo, azul marino profundo, burdeos, blanco roto/marfil/crudo.
- REGLA DE COMPENSACION PARA NEGRO Y GRIS FRIO: Nunca debe llevar negro puro ni gris frio pegados directamente a la cara. Si una prenda superior (jersey, sudadera, sobrecamisa, cazadora cerrada) es negra o gris fria, ES OBLIGATORIO que lleve debajo una camiseta basica en blanco roto/marfil asomando claramente por el cuello para crear una barrera de luz calida. En pantalones y calzado si se acepta negro lavado o sastre sin problema.
- COLORES PROHIBIDOS: Blanco optico nuclear, colores pastel frios y tonos fluor/neon.

3. ESTILO ("PIJO CON ROLLO" / SMART-CASUAL URBANO):
- SILUETA: Cortes rectos (regular/straight fit) o ligeramente amplios/estructurados (tailored/boxy) que equilibren su espalda y hombros atleticos. PROHIBIDO el corte pitillo (skinny) o prendas muy apretadas.
- CODIGOS: Cero logos grandes/llamativos y cero camisas formales clasicas de botones para el dia a dia.
- PRENDAS BASE: Vaqueros rectos (azul claro, azul medio, negro lavado), pantalones sastre/pinzas relajados, jerseis de punto con textura (acanalados, trenzados, media cremallera), cazadoras cortas estructuradas, camisetas basicas de buen gramaje en blanco roto.
- CALZADO HABITUAL: Zapatillas limpias de perfil bajo/retro (Adidas Samba, Coolway Olive, Gazelle), Birkenstock y mocasines negros de piel con suela algo gruesa.

4. REGLA DE TRADUCCION DE REFERENCIAS DE ESTILO:
- Cuando analices fotos de outfits de otras personas, IGNORA el tono de piel y los colores originales del modelo como requisito (por ejemplo, si el modelo es una persona negra vestida entera de negro, no rechaces el look ni le obligues a vestir de negro).
- Extrae exclusivamente el ESQUELETO DEL ESTILO (cortes, volumenes, juego de capas, caida de la ropa y nivel de elegancia).
- Evalua como funcionaria esa silueta en un cuerpo de 1,83 m y 85 kg con hombros anchos, y TRADUCE los colores originales a la paleta Otono Calido y prendas del cliente.

5. ACTITUD Y TONO:
- Se directo, critico y pragmatico. Nada de cumplidos vacios: si una prenda le apaga la cara, le acorta el torso o repite lo que ya tiene, dilo sin rodeos.
"""

# ==============================================================================
# 3. ESQUEMAS DE RESPUESTA ESTRUCTURADA (PYDANTIC)
# ==============================================================================
class PrendaAnalisis(BaseModel):
    categoria: Literal["Arriba", "Chaqueta", "Pantalon", "Calzado"]
    nombre_corto: str = Field(
        description="Nombre descriptivo breve, ej: 'Jersey marron chocolate acanalado'."
    )
    color_principal: str
    temperatura_cromatica: Literal["Calida", "Neutra calida", "Neutra fria", "Fria"]
    requiere_compensacion_cuello: bool = Field(
        description="True SOLO si es una prenda de 'Arriba' o 'Chaqueta' negra o gris fria que necesita camiseta marfil asomando."
    )
    nivel_abrigo: Literal["Ligero", "Medio", "Alto"]
    elegancia: int = Field(ge=1, le=10)
    notas_estilismo: str


class ReferenciaAnalisis(BaseModel):
    elegancia: int = Field(
        ge=1,
        le=10,
        description="Nivel de elegancia del 1 al 10 calibrado a su escala (3=gym/casa, 5=FIDIS/facultad, 7=cena arreglada sin traje, 9=formal).",
    )
    titulo_estilo: str = Field(
        description="Nombre corto del estilo, ej: 'Cazadora corta estructurada + pantalon recto y mocasin'."
    )
    esqueleto_estilo: str = Field(
        description="Receta estructural pura (cortes, volumenes, capas y proporciones), ignorando los colores."
    )
    evaluacion_silueta: str = Field(
        description="Como le quedaria exactamente este corte y proporcion a un hombre de 1,83 m y 85 kg de espalda ancha."
    )
    traduccion_otono_calido: str = Field(
        description="Como adaptar este outfit a su colorimetria Otono Calido (que tonos cambiar arriba y que mantener abajo)."
    )
    veredicto_adaptacion: Literal[
        "FAVORECE TAL CUAL",
        "FAVORECE ADAPTANDO COLORES",
        "ADAPTAR CORTE Y COLORES",
    ]


class OutfitPropuesta(BaseModel):
    titulo: str
    referencia_inspiracion: str = Field(
        description="Codigo de la referencia de estilo en la que se basa (ej. 'REF-002') o 'Diseno propio segun ocasion'."
    )
    ids_prendas: list[str] = Field(
        description="Lista exacta de IDs de prendas del inventario, ej: ['#001', '#005', '#011']."
    )
    regla_capas_y_cuello: str = Field(
        description="Como combinar capas y si debe asomar camiseta blanco roto por el cuello."
    )
    explicacion_estilista: str = Field(
        description="Por que favorece a su fisico (1,83m/85kg), colorimetria Otono Calido y clima."
    )


class RespuestaOutfits(BaseModel):
    outfits: list[OutfitPropuesta] = Field(min_length=3, max_length=3)


class EvaluacionCompra(BaseModel):
    nota: int = Field(ge=1, le=10)
    veredicto: Literal["COMPRAR", "SOLO SI ES CHOLLO", "DESCARTAR"]
    analisis_colorimetria_y_corte: str
    analisis_redundancia: str
    encaje_con_referencias: str = Field(
        description="Si esta prenda ayuda a recrear los estilos guardados en sus referencias."
    )
    ids_combinables: list[str]
    ejemplo_outfit_completo: str


# ==============================================================================
# 4. CONEXION A SUPABASE Y GEMINI + UTILIDADES DE IMAGEN
# ==============================================================================
def obtener_secret(clave: str) -> str:
    if clave in st.secrets:
        return str(st.secrets[clave])
    return os.environ.get(clave, "")


@st.cache_resource
def obtener_supabase() -> Client:
    url = obtener_secret("SUPABASE_URL").strip()
    key = obtener_secret("SUPABASE_KEY").strip()
    return create_client(url, key)


@st.cache_resource
def obtener_gemini() -> genai.Client:
    api_key = obtener_secret("GEMINI_API_KEY").strip()
    if api_key.startswith("AQ."):
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)


def normalizar_id(raw_id: str) -> str:
    digitos = re.sub(r"\D", "", str(raw_id))
    return f"#{int(digitos):03d}" if digitos else str(raw_id)


def limpiar_nombre_archivo(nombre: str) -> str:
    stem = Path(nombre).stem
    limpio = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return f"{limpio}.jpg"


def preparar_imagen(fuente) -> tuple[bytes, Image.Image]:
    img = Image.open(fuente)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "P", "LA"):
        fondo = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        fondo.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
        img = fondo
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue(), img


def siguiente_codigo(filas: list[dict], prefijo: str) -> str:
    max_num = 0
    for fila in filas:
        digitos = re.sub(r"\D", "", fila.get("codigo", ""))
        if digitos:
            max_num = max(max_num, int(digitos))
    return f"{prefijo}{max_num + 1:03d}"


def extraer_nota_nombre_archivo(nombre: str) -> int | None:
    match = re.match(r"^([1-9]|10)[_-]", nombre)
    if match:
        return int(match.group(1))
    return None


# ==============================================================================
# 5. SINCRONIZACION ESCRITORIO -> SUPABASE (SOLO EN MODO LOCAL PC)
# ==============================================================================
def sincronizar_carpeta_armario(supabase: Client, gemini: genai.Client) -> tuple[int, int, list[str]]:
    archivos_locales = {
        f.name: f
        for f in sorted(CARPETA_ARMARIO.iterdir())
        if f.is_file() and f.suffix.lower() in EXTENSIONES_VALIDAS
    }
    filas_bd = supabase.table("prendas").select("*").execute().data
    por_nombre = {f["archivo_nombre"]: f for f in filas_bd}

    eliminadas = 0
    for nombre_bd, fila in por_nombre.items():
        if nombre_bd not in archivos_locales:
            try:
                supabase.storage.from_(BUCKET_NAME).remove([fila["storage_path"]])
            except Exception:
                pass
            supabase.table("prendas").delete().eq("id", fila["id"]).execute()
            eliminadas += 1

    nuevas = [n for n in archivos_locales if n not in por_nombre]
    errores = []
    anadidas = 0

    if nuevas:
        filas_actuales = [f for n, f in por_nombre.items() if n in archivos_locales]
        barra = st.progress(0, text="Sincronizando ropa nueva con Supabase y Gemini...")
        for idx, nombre_archivo in enumerate(nuevas):
            barra.progress(
                (idx + 1) / len(nuevas),
                text=f"Analizando prenda ({idx + 1}/{len(nuevas)}): {nombre_archivo}",
            )
            try:
                jpeg_bytes, img_pil = preparar_imagen(archivos_locales[nombre_archivo])
                storage_path = f"prendas/{limpiar_nombre_archivo(nombre_archivo)}"
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=jpeg_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
                url_publica = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path).rstrip("?")

                prompt = (
                    "Cataloga esta prenda del armario del cliente segun su colorimetria Otono Calido, "
                    "complexion (1,83 m, 85 kg) y escala de elegancia del 1 al 10."
                )
                resp = gemini.models.generate_content(
                    model=MODELO_GEMINI,
                    contents=[img_pil, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT_ESTILISTA,
                        response_mime_type="application/json",
                        response_schema=PrendaAnalisis,
                        temperature=0.2,
                    ),
                )
                datos = json.loads(resp.text)
                codigo = siguiente_codigo(filas_actuales, "#")

                nuevo_registro = {
                    "codigo": codigo,
                    "archivo_nombre": nombre_archivo,
                    "storage_path": storage_path,
                    "url_imagen": url_publica,
                    "disponible": True,
                    **datos,
                }
                insertado = supabase.table("prendas").insert(nuevo_registro).execute().data[0]
                filas_actuales.append(insertado)
                anadidas += 1
            except Exception as exc:
                errores.append(f"{nombre_archivo}: {exc}")
        barra.empty()

    return anadidas, eliminadas, errores


def sincronizar_carpeta_referencias(
    supabase: Client, gemini: genai.Client
) -> tuple[int, int, list[str]]:
    archivos_locales = {
        f.name: f
        for f in sorted(CARPETA_REFERENCIAS.iterdir())
        if f.is_file() and f.suffix.lower() in EXTENSIONES_VALIDAS
    }
    filas_bd = supabase.table("referencias_estilo").select("*").execute().data
    por_nombre = {f["archivo_nombre"]: f for f in filas_bd}

    eliminadas = 0
    for nombre_bd, fila in por_nombre.items():
        if nombre_bd not in archivos_locales:
            try:
                supabase.storage.from_(BUCKET_NAME).remove([fila["storage_path"]])
            except Exception:
                pass
            supabase.table("referencias_estilo").delete().eq("id", fila["id"]).execute()
            eliminadas += 1

    nuevas = [n for n in archivos_locales if n not in por_nombre]
    errores = []
    anadidas = 0

    if nuevas:
        filas_actuales = [f for n, f in por_nombre.items() if n in archivos_locales]
        barra = st.progress(0, text="Analizando outfits de referencia...")
        for idx, nombre_archivo in enumerate(nuevas):
            barra.progress(
                (idx + 1) / len(nuevas),
                text=f"Traduciendo estilo ({idx + 1}/{len(nuevas)}): {nombre_archivo}",
            )
            try:
                jpeg_bytes, img_pil = preparar_imagen(archivos_locales[nombre_archivo])
                storage_path = f"referencias/{limpiar_nombre_archivo(nombre_archivo)}"
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=jpeg_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
                url_publica = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path).rstrip("?")

                nota_prefijo = extraer_nota_nombre_archivo(nombre_archivo)
                instruccion_nota = (
                    f"El cliente ha pre-etiquetado este outfit con nivel de elegancia {nota_prefijo}/10; respeta ese valor en 'elegancia'."
                    if nota_prefijo
                    else "Asigna tu la nota de elegancia del 1 al 10 segun su escala personal."
                )

                prompt = f"""
                Analiza este outfit de referencia que le ha gustado al cliente.
                IMPORTANTE:
                - NO evalues si los colores de la foto le quedan bien al modelo de la foto; extrae unicamente el ESQUELETO DEL ESTILO (cortes, proporciones, capas y rollo).
                - Evalua como le quedaria esa silueta al cliente (1,83 m, 85 kg, espalda ancha) y TRADUCE los colores originales a su paleta Otono Calido (por ejemplo, si el modelo lleva negro puro arriba, indica cambiarlo por marron chocolate, marino profundo u oliva con camiseta blanco roto al cuello).
                - {instruccion_nota}
                """
                resp = gemini.models.generate_content(
                    model=MODELO_GEMINI,
                    contents=[img_pil, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT_ESTILISTA,
                        response_mime_type="application/json",
                        response_schema=ReferenciaAnalisis,
                        temperature=0.25,
                    ),
                )
                datos = json.loads(resp.text)
                if nota_prefijo:
                    datos["elegancia"] = nota_prefijo

                codigo = siguiente_codigo(filas_actuales, "REF-")
                nuevo_registro = {
                    "codigo": codigo,
                    "archivo_nombre": nombre_archivo,
                    "storage_path": storage_path,
                    "url_imagen": url_publica,
                    **datos,
                }
                insertado = (
                    supabase.table("referencias_estilo").insert(nuevo_registro).execute().data[0]
                )
                filas_actuales.append(insertado)
                anadidas += 1
            except Exception as exc:
                errores.append(f"{nombre_archivo}: {exc}")
        barra.empty()

    return anadidas, eliminadas, errores


# ==============================================================================
# 6. INTELIGENCIA DE OUTFITS Y AUDITORIA DE COMPRAS
# ==============================================================================
def generar_outfits_ia(
    gemini: genai.Client,
    prendas_activas: list[dict],
    referencias: list[dict],
    elegancia: int,
    clima: str,
    peticion_extra: str,
) -> RespuestaOutfits:
    catalogo_ropa = [
        {
            "id": p["codigo"],
            "categoria": p["categoria"],
            "prenda": p["nombre_corto"],
            "color": p["color_principal"],
            "temperatura": p["temperatura_cromatica"],
            "requiere_camiseta_marfil_cuello": p["requiere_compensacion_cuello"],
            "abrigo": p["nivel_abrigo"],
            "elegancia": p["elegancia"],
        }
        for p in prendas_activas
    ]

    catalogo_refs = [
        {
            "id": r["codigo"],
            "elegancia": r["elegancia"],
            "titulo": r["titulo_estilo"],
            "esqueleto": r["esqueleto_estilo"],
            "traduccion_color": r["traduccion_otono_calido"],
        }
        for r in referencias
    ]

    prompt = f"""
    Disena 3 outfits completos y distintos usando EXCLUSIVAMENTE prendas disponibles del inventario del cliente.

    INVENTARIO DISPONIBLE HOY:
    {json.dumps(catalogo_ropa, ensure_ascii=False)}

    REFERENCIAS DE ESTILO GUARDADAS POR EL CLIENTE (Usalas como guia de cortes, proporciones y calibracion de elegancia, aplicando siempre su traduccion a Otono Calido):
    {json.dumps(catalogo_refs, ensure_ascii=False)}

    PARAMETROS DE HOY:
    - Nivel de elegancia buscado: {elegancia}/10
      (3/10 = gimnasio/casa/recados; 5/10 = dia a dia en centro de investigacion FIDIS en CHUS o facultad en Santiago; 7/10 = comida/cena arreglada "mayado" sin traje; 9/10 = evento muy formal).
    - Clima: {clima}
    - Condicion adicional del cliente: "{peticion_extra if peticion_extra.strip() else 'Ninguna restriccion extra.'}"

    REGLAS ESTRICTAS:
    - Usa los codigos exactos (ej. '#001').
    - Si una prenda superior es negra o gris fria, exige en 'regla_capas_y_cuello' llevar camiseta blanco roto/marfil asomando por el cuello.
    """

    resp = gemini.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_ESTILISTA,
            response_mime_type="application/json",
            response_schema=RespuestaOutfits,
            temperature=0.6,
        ),
    )
    return RespuestaOutfits.model_validate_json(resp.text)


def evaluar_compra_ia(
    gemini: genai.Client,
    img_pil: Image.Image,
    prendas: list[dict],
    referencias: list[dict],
    notas_usuario: str,
) -> EvaluacionCompra:
    catalogo_ropa = [
        {
            "id": p["codigo"],
            "categoria": p["categoria"],
            "prenda": p["nombre_corto"],
            "color": p["color_principal"],
            "elegancia": p["elegancia"],
        }
        for p in prendas
    ]
    catalogo_refs = [
        {
            "id": r["codigo"],
            "elegancia": r["elegancia"],
            "titulo": r["titulo_estilo"],
            "esqueleto": r["esqueleto_estilo"],
        }
        for r in referencias
    ]

    prompt = f"""
    Evalua la prenda de la foto como posible compra nueva para el cliente.
    Datos aportados por el cliente: "{notas_usuario if notas_usuario.strip() else 'Ninguno.'}"

    SU ARMARIO ACTUAL:
    {json.dumps(catalogo_ropa, ensure_ascii=False)}

    SUS REFERENCIAS DE ESTILO:
    {json.dumps(catalogo_refs, ensure_ascii=False)}

    Analiza sin filtros:
    1. Colorimetria Otono Calido y corte para su fisico (1,83 m, 85 kg, espalda ancha).
    2. Repite algo que ya tiene en su armario o cubre un hueco real?
    3. Encaja con los estilos de sus referencias guardadas?
    4. Con que IDs exactos de su armario actual combina mejor?
    """

    resp = gemini.models.generate_content(
        model=MODELO_GEMINI,
        contents=[img_pil, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_ESTILISTA,
            response_mime_type="application/json",
            response_schema=EvaluacionCompra,
            temperature=0.3,
        ),
    )
    return EvaluacionCompra.model_validate_json(resp.text)


# ==============================================================================
# 7. INTERFAZ VISUAL (OPTIMIZADA PARA ORDENADOR Y MOVIL)
# ==============================================================================
def configurar_estilos() -> None:
    st.set_page_config(
        page_title="Estilista IA · Otono Calido",
        page_icon="🧥",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0a0a0f;
            color: #f1f1f4;
        }
        .badge-warm {
            background-color: #3d2817;
            color: #f5b971;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-cool {
            background-color: #2c1d24;
            color: #f28b82;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-neutral {
            background-color: #1e2433;
            color: #9bb4e8;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def comprobar_acceso() -> bool:
    password_real = obtener_secret("APP_PASSWORD")
    if not password_real:
        return True
    if st.session_state.get("autenticado", False):
        return True

    st.title("🔒 Vestidor Privado IA")
    pwd = st.text_input("Introduce tu contrasena de acceso:", type="password")
    if st.button("Entrar", type="primary"):
        if pwd == password_real:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Contrasena incorrecta.")
    return False


def describir_elegancia(nivel: int) -> str:
    if nivel <= 3:
        return "3/10 · Gimnasio, estar por casa o recados informales"
    if nivel <= 5:
        return "5/10 · Dia a dia (FIDIS Hospital CHUS / Universidad Santiago)"
    if nivel <= 7:
        return "7/10 · Comida elegante, cena arreglada o cita ('mayado' sin traje)"
    return "9/10 · Evento muy formal"


def main() -> None:
    configurar_estilos()
    if not comprobar_acceso():
        return

    supabase = obtener_supabase()
    gemini = obtener_gemini()

    prendas = supabase.table("prendas").select("*").order("codigo").execute().data
    referencias = supabase.table("referencias_estilo").select("*").order("codigo").execute().data
    mapa_prendas = {normalizar_id(p["codigo"]): p for p in prendas}

    with st.sidebar:
        st.title("🧥 Estilista Personal IA")
        st.caption("Otono Calido · 1,83 m / 85 kg · Galicia")

        if EN_NUBE:
            st.info("📱 **Modo Movil / Nube activo.** La sincronizacion de carpetas se realiza desde el ordenador.")
        else:
            st.success("💻 **Modo Ordenador activo.** Carpetas del Escritorio conectadas.")
            if st.button("🔄 Sincronizar Todo con Nube", type="primary", use_container_width=True):
                add_p, del_p, err_p = sincronizar_carpeta_armario(supabase, gemini)
                add_r, del_r, err_r = sincronizar_carpeta_referencias(supabase, gemini)
                st.success(
                    f"Armario: +{add_p} / -{del_p} · Referencias: +{add_r} / -{del_r}"
                )
                for e in err_p + err_r:
                    st.warning(e)
                st.rerun()

        st.divider()
        disponibles_count = sum(1 for p in prendas if p.get("disponible", True))
        st.metric("Prendas disponibles", f"{disponibles_count} / {len(prendas)}")
        st.metric("Outfits de referencia", len(referencias))

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1️⃣ Mi Armario",
            "2️⃣ Referencias de Estilo",
            "3️⃣ Generar Outfits",
            "4️⃣ ¿Me la compro?",
        ]
    )

    with tab1:
        st.subheader("Tu Armario Sincronizado")
        if not prendas:
            st.info(
                "Aun no hay prendas. Mete tus fotos en `Estilista_IA/mi_armario` en el ordenador y pulsa **Sincronizar Todo con Nube**."
            )
        else:
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                cat_sel = st.selectbox(
                    "Categoria", ["Todas", "Arriba", "Chaqueta", "Pantalon", "Calzado"]
                )
            with c2:
                estado_sel = st.selectbox(
                    "Disponibilidad", ["Solo disponibles", "Todas", "En lavadora / No disponibles"]
                )
            with c3:
                eleg_min = st.slider("Elegancia minima", 1, 10, 1)

            filtradas = prendas
            if cat_sel != "Todas":
                filtradas = [p for p in filtradas if p["categoria"] == cat_sel]
            if estado_sel == "Solo disponibles":
                filtradas = [p for p in filtradas if p.get("disponible", True)]
            elif estado_sel == "En lavadora / No disponibles":
                filtradas = [p for p in filtradas if not p.get("disponible", True)]
            filtradas = [p for p in filtradas if p["elegancia"] >= eleg_min]

            cols = st.columns(4)
            for idx, p in enumerate(filtradas):
                with cols[idx % 4]:
                    st.image(p["url_imagen"], use_container_width=True)
                    st.markdown(f"**{p['codigo']} · {p['nombre_corto']}**")

                    temp = p["temperatura_cromatica"]
                    cls = (
                        "badge-warm"
                        if "Calida" in temp
                        else ("badge-cool" if "Fria" in temp else "badge-neutral")
                    )
                    st.markdown(
                        f"<span class='{cls}'>{temp}</span> · 🧥 {p['nivel_abrigo']} · ⭐ **{p['elegancia']}/10**",
                        unsafe_allow_html=True,
                    )
                    if p.get("requiere_compensacion_cuello"):
                        st.caption("⚠️ Usar con camiseta marfil asomando al cuello.")

                    disp_actual = p.get("disponible", True)
                    etiqueta_btn = "🟢 Disponible" if disp_actual else "🧺 En lavadora"
                    if st.button(etiqueta_btn, key=f"disp_{p['id']}", use_container_width=True):
                        supabase.table("prendas").update({"disponible": not disp_actual}).eq(
                            "id", p["id"]
                        ).execute()
                        st.rerun()

    with tab2:
        st.subheader("Outfits de Referencia (Traducidos a tu Fisico y Colorimetria)")
        st.caption(
            "La IA extrae unicamente el estilo, cortes y proporciones de estas fotos, y adapta los colores a tu paleta Otono Calido."
        )
        if not referencias:
            st.info(
                "No hay referencias aun. Guarda fotos de outfits que te gusten en `Estilista_IA/referencias_estilo` y sincroniza desde el ordenador."
            )
        else:
            filtro_ref_eleg = st.slider(
                "Filtrar referencias por nivel de elegancia", 1, 10, (1, 10)
            )
            refs_filtradas = [
                r
                for r in referencias
                if filtro_ref_eleg[0] <= r["elegancia"] <= filtro_ref_eleg[1]
            ]

            for r in refs_filtradas:
                st.markdown("---")
                col_img, col_txt = st.columns([1, 2.5])
                with col_img:
                    st.image(r["url_imagen"], use_container_width=True)
                    nueva_nota = st.number_input(
                        f"Nota elegancia ({r['codigo']})",
                        min_value=1,
                        max_value=10,
                        value=int(r["elegancia"]),
                        key=f"nota_{r['id']}",
                    )
                    if nueva_nota != int(r["elegancia"]):
                        supabase.table("referencias_estilo").update(
                            {"elegancia": int(nueva_nota)}
                        ).eq("id", r["id"]).execute()
                        st.rerun()

                with col_txt:
                    st.markdown(f"### {r['codigo']} — {r['titulo_estilo']}")
                    st.markdown(
                        f"**Elegancia:** ⭐ `{r['elegancia']}/10` · **Diagnostico:** `{r['veredicto_adaptacion']}`"
                    )
                    st.markdown(f"**📐 Esqueleto del estilo (cortes y capas):** {r['esqueleto_estilo']}")
                    st.markdown(f"**🏋️ Como te queda a ti (1,83 m / 85 kg):** {r['evaluacion_silueta']}")
                    st.info(f"**🎨 Traduccion a Otono Calido:** {r['traduccion_otono_calido']}")

    with tab3:
        st.subheader("Generador de Outfits Calibrado a tu Estilo")
        prendas_disponibles = [p for p in prendas if p.get("disponible", True)]

        if len(prendas_disponibles) < 3:
            st.warning("Necesitas al menos 3 prendas disponibles para generar combinaciones.")
        else:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                nivel_eleg = st.slider("Nivel de Elegancia (1 a 10)", 1, 10, 5)
                st.info(f"Ocasion: **{describir_elegancia(nivel_eleg)}**")
            with col_b:
                clima_sel = st.selectbox(
                    "Clima",
                    [
                        "Frio / Lluvia (Galicia Otono-Invierno)",
                        "Entretiempo (Fresco suave)",
                        "Calor (Verano)",
                    ],
                )

            extra_txt = st.text_input(
                "Condicion opcional",
                placeholder="Ej. 'Quiero llevar los mocasines negros' o 'Con vaquero negro lavado'",
            )

            if st.button("✨ Generar 3 Outfits", type="primary"):
                with st.spinner("Combinando tus prendas segun tus referencias de estilo..."):
                    res = generar_outfits_ia(
                        gemini, prendas_disponibles, referencias, nivel_eleg, clima_sel, extra_txt
                    )
                    st.session_state["outfits_generados"] = res.outfits

            if "outfits_generados" in st.session_state:
                for idx, outfit in enumerate(st.session_state["outfits_generados"], start=1):
                    st.markdown("---")
                    st.markdown(f"### Propuesta #{idx}: {outfit.titulo}")
                    st.caption(f"Inspiracion base: **{outfit.referencia_inspiracion}**")

                    ids_norm = [normalizar_id(i) for i in outfit.ids_prendas]
                    items_look = [mapa_prendas[i] for i in ids_norm if i in mapa_prendas]

                    if items_look:
                        img_cols = st.columns(len(items_look))
                        for c_img, item in zip(img_cols, items_look):
                            with c_img:
                                st.image(item["url_imagen"], use_container_width=True)
                                st.caption(f"**{item['codigo']}** · {item['nombre_corto']}")

                    st.success(f"👕 **Capas y cuello:** {outfit.regla_capas_y_cuello}")
                    st.write(f"🧠 **Analisis:** {outfit.explicacion_estilista}")

    with tab4:
        st.subheader("Auditoria de Compra en Tienda u Online")
        st.caption(
            "Hazle una foto con el movil en la tienda o sube una captura. La IA comprobara si te favorece, si encaja con tus referencias y si repite ropa que ya tienes."
        )

        modo_foto = st.radio(
            "Como quieres cargar la prenda?",
            ["🖼️ Subir foto o captura", "📸 Usar camara del movil ahora"],
            horizontal=True,
        )

        archivo_foto = None
        if modo_foto == "🖼️ Subir foto o captura":
            archivo_foto = st.file_uploader(
                "Selecciona imagen", type=["jpg", "jpeg", "png", "webp"]
            )
        else:
            archivo_foto = st.camera_input("Haz foto a la prenda")

        notas_compra = st.text_input(
            "Detalles opcionales (precio, tejido, corte)",
            placeholder="Ej. 'Cazadora Massimo Dutti 89€, corte boxy, algodon grueso'",
        )

        if archivo_foto is not None:
            _, img_candidata = preparar_imagen(archivo_foto)
            col_izq, col_der = st.columns([1, 2])
            with col_izq:
                st.image(img_candidata, caption="Prenda candidata", use_container_width=True)
                btn_auditar = st.button("⚖️ Auditar Compra", type="primary", use_container_width=True)

            with col_der:
                if btn_auditar:
                    with st.spinner("Auditando colorimetria, silueta, armario y referencias..."):
                        ev = evaluar_compra_ia(
                            gemini, img_candidata, prendas, referencias, notas_compra
                        )
                        st.session_state["auditoria_compra"] = ev

                if "auditoria_compra" in st.session_state:
                    ev: EvaluacionCompra = st.session_state["auditoria_compra"]
                    if ev.veredicto == "COMPRAR":
                        st.success(f"### Veredicto: {ev.veredicto} · Nota: {ev.nota}/10")
                    elif ev.veredicto == "SOLO SI ES CHOLLO":
                        st.warning(f"### Veredicto: {ev.veredicto} · Nota: {ev.nota}/10")
                    else:
                        st.error(f"### Veredicto: {ev.veredicto} · Nota: {ev.nota}/10")

                    st.markdown(f"**🎨 Colorimetria y corte:** {ev.analisis_colorimetria_y_corte}")
                    st.markdown(f"**📦 Repite tu armario?:** {ev.analisis_redundancia}")
                    st.markdown(f"**📐 Encaje con tus referencias:** {ev.encaje_con_referencias}")
                    st.info(f"**👔 Como estrenarla:** {ev.ejemplo_outfit_completo}")

                    if ev.ids_combinables:
                        st.markdown("**Prendas exactas de tu armario que combinan con ella:**")
                        ids_comb = [normalizar_id(i) for i in ev.ids_combinables]
                        comb_items = [mapa_prendas[i] for i in ids_comb if i in mapa_prendas]
                        if comb_items:
                            c_cols = st.columns(min(len(comb_items), 4))
                            for i_c, item_c in enumerate(comb_items):
                                with c_cols[i_c % 4]:
                                    st.image(item_c["url_imagen"], use_container_width=True)
                                    st.caption(f"**{item_c['codigo']}** · {item_c['nombre_corto']}")


if __name__ == "__main__":
    main()
