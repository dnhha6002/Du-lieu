# python.py
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# ******************************* KHỞI TẠO STATE VÀ HÀM CHAT (Đã giữ nguyên) *******************************

if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "df_processed_for_chat" not in st.session_state:
    st.session_state.df_processed_for_chat = None
if "data_context_markdown" not in st.session_state:
    st.session_state.data_context_markdown = ""

def chat_with_gemini(prompt, data_context, api_key, history):
    """Xử lý tin nhắn chat, gửi nội dung và lịch sử đến Gemini với ngữ cảnh dữ liệu."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # Thêm ngữ cảnh dữ liệu vào tin nhắn đầu tiên của user trong chuỗi hội thoại
        if not [msg for msg in history if msg['role'] == 'assistant']: # Kiểm tra nếu đây là tin nhắn đầu tiên trong lịch sử chat
             context_prompt = f"""
            Bạn là một trợ lý phân tích tài chính chuyên nghiệp. Hãy trả lời các câu hỏi của người dùng dựa trên ngữ cảnh dữ liệu sau.
            Ngữ cảnh dữ liệu tài chính đã xử lý (Tốc độ tăng trưởng và Tỷ trọng):
            {data_context}
            
            Đây là câu hỏi của người dùng: {prompt}
            """
        else:
            context_prompt = prompt
        
        # Chuẩn bị lịch sử cho Gemini API
        gemini_history = []
        for message in history:
            gemini_history.append(
                genai.types.Content(
                    role="user" if message["role"] == "user" else "model",
                    parts=[genai.types.Part.from_text(message["content"])]
                )
            )

        # Thêm tin nhắn user hiện tại (có chứa context hoặc không)
        gemini_history.append(
            genai.types.Content(
                role="user", 
                parts=[genai.types.Part.from_text(context_prompt)]
            )
        )
            
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_history
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Lỗi {e.status_code}. Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# ******************************* KẾT THÚC PHẦN KHỞI TẠO STATE VÀ HÀM CHAT *******************************


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) --- 
@st.cache_data 
def process_financial_data(df): 
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng.""" 
    # [Giữ nguyên code hàm process_financial_data]
    numeric_cols = ['Năm trước', 'Năm sau'] 
    for col in numeric_cols: 
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
    
    # 1. Tính Tốc độ Tăng trưởng 
    df['Tốc độ tăng trưởng (%)'] = ( 
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9) 
    ) * 100 

    # 2. Tính Tỷ trọng theo Tổng Tài sản 
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)] 
    
    if tong_tai_san_row.empty: 
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.") 

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0] 
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0] 

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU ******************************* # **Đảm bảo divisor_N_1 và divisor_N luôn được định nghĩa trong hàm này**
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9 
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9 

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100 
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100 
    # ******************************* PHẦN SỬA LỖI KẾT THÚC ******************************* return df 

# --- Hàm gọi API Gemini --- 
def get_ai_analysis(data_for_ai, api_key): 
    # [Giữ nguyên code hàm get_ai_analysis]
    try: 
        client = genai.Client(api_key=api_key) 
        model_name = 'gemini-2.5-flash' 

        prompt = f""" 
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành. 
        
        Dữ liệu thô và chỉ số: 
        {data_for_ai} 
        """ 

        response = client.models.generate_content( 
            model=model_name, 
            contents=prompt 
        ) 
        return response.text 

    except APIError as e: 
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}" 
    except KeyError: 
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud." 
    except Exception as e: 
        return f"Đã xảy ra lỗi không xác định: {e}" 


# --- Chức năng 1: Tải File --- 
uploaded_file = st.file_uploader( 
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)", 
    type=['xlsx', 'xls'] 
) 

if uploaded_file is not None: 
    # ******************************* KHẮC PHỤC LỖI #1: Khởi tạo biến THH *******************************
    # Phải khởi tạo các biến chỉ số tài chính bên ngoài khối try/except nhỏ hơn 
    # để chúng có thể được truy cập an toàn ở Chức năng 5 và Chức năng 6.
    thanh_toan_hien_hanh_N = "N/A"
    thanh_toan_hien_hanh_N_1 = "N/A"

    try: 
        df_raw = pd.read_excel(uploaded_file) 
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng 
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau'] 
        
        # Xử lý dữ liệu 
        df_processed = process_financial_data(df_raw.copy()) 

        # Đặt lại state cho Chat khi file mới được tải lên
        st.session_state.df_processed_for_chat = df_processed.copy()
        st.session_state.messages = []
        
        if df_processed is not None: 
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả --- 
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản") 
            st.dataframe(df_processed.style.format({ 
                'Năm trước': '{:,.0f}', 
                'Năm sau': '{:,.0f}', 
                'Tốc độ tăng trưởng (%)': '{:.2f}%', 
                'Tỷ trọng Năm trước (%)': '{:.2f}%', 
                'Tỷ trọng Năm sau (%)': '{:.2f}%' 
            }), use_container_width=True) 
            
            # --- Chức năng 4: Tính Chỉ số Tài chính --- 
            st.subheader("4. Các Chỉ số Tài chính Cơ bản") 
            
            try: 
                # Lấy Tài sản ngắn hạn 
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0] 
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0] 

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0] 

                # Tính toán (Xử lý chia cho 0)
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')

                col1, col2 = st.columns(2) 
                with col1: 
                    st.metric( 
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)", 
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "Vô hạn"
                    ) 
                with col2: 
                    st.metric( 
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)", 
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "Vô hạn",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf') else None
                    ) 
                    
            except IndexError: 
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.") 
                # Giữ nguyên giá trị "N/A" đã khởi tạo ở trên
            except ZeroDivisionError:
                st.warning("Lỗi: Nợ ngắn hạn bằng 0, chỉ số thanh toán hiện hành là vô hạn.")
                thanh_toan_hien_hanh_N = float('inf')
                thanh_toan_hien_hanh_N_1 = float('inf')

            
            # --- Chức năng 5: Nhận xét AI --- 
            
            # Chuẩn bị dữ liệu để gửi cho AI (cho cả Chức năng 5 và Khung chat)
            
            # Lấy tốc độ tăng trưởng TSNH 
            try:
                tsnh_tang_truong = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]
            except IndexError:
                tsnh_tang_truong = "N/A"
            
            # Chuyển đổi các chỉ số sang chuỗi để đưa vào Markdown
            thh_n_str = f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, (float, int)) and thanh_toan_hien_hanh_N != float('inf') else str(thanh_toan_hien_hanh_N)
            thh_n_1_str = f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, (float, int)) and thanh_toan_hien_hanh_N_1 != float('inf') else str(thanh_toan_hien_hanh_N_1)
            
            data_for_ai_df = pd.DataFrame({ 
                'Chỉ tiêu': [ 
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)' 
                ], 
                'Giá trị': [ 
                    df_processed.to_markdown(index=False), 
                    f"{tsnh_tang_truong:.2f}%" if isinstance(tsnh_tang_truong, (float, int)) else tsnh_tang_truong, 
                    thh_n_1_str, 
                    thh_n_str
                ] 
            }) 

            # Lưu ngữ cảnh vào Session State để hàm Chat có thể sử dụng
            st.session_state.data_context_markdown = data_for_ai_df.to_markdown(index=False)
            
            # Tiếp tục Chức năng 5
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)") 
            if st.button("Yêu cầu AI Phân tích"): 
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key: 
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'): 
                        ai_result = get_ai_analysis(st.session_state.data_context_markdown, api_key) 
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**") 
                        st.info(ai_result) 
                else: 
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.") 
            
            # ******************************* KHUNG CHAT (CHỨC NĂNG 6) *******************************
            st.markdown("---")
            st.subheader("6. Chat với Gemini AI để Phân tích chuyên sâu 💬")
            st.caption("AI sẽ nhớ lịch sử chat và có ngữ cảnh là Báo cáo tài chính bạn vừa tải lên.")

            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Xử lý input chat
            if prompt := st.chat_input("Bạn muốn hỏi gì về Báo cáo tài chính này? (VD: Đánh giá cơ cấu tài sản)"):
                
                # Thêm tin nhắn của user vào lịch sử và hiển thị
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                api_key = st.secrets.get("GEMINI_API_KEY") 

                if not api_key:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
                elif st.session_state.df_processed_for_chat is None:
                     st.error("Lỗi: Không tìm thấy dữ liệu tài chính đã xử lý. Vui lòng tải lại file.")
                else:
                    with st.chat_message("assistant"):
                        with st.spinner("Đang chờ Gemini phản hồi..."):
                            full_response = chat_with_gemini(
                                prompt=prompt, 
                                data_context=st.session_state.data_context_markdown, # Ngữ cảnh dữ liệu đã lưu
                                api_key=api_key,
                                history=st.session_state.messages[:-1] # Lịch sử trừ tin nhắn mới nhất
                            )
                            
                            st.markdown(full_response)
                        
                        # Thêm phản hồi của AI vào lịch sử chat
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

            # ******************************* KẾT THÚC KHUNG CHAT *******************************
            
    except ValueError as ve: 
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}") 
    # ******************************* KHẮC PHỤC LỖI #2: Loại bỏ tham chiếu biến chưa định nghĩa *******************************
    # Đảm bảo khối except này chỉ bắt những lỗi không phải là lỗi định nghĩa biến 
    # do những biến đó (như divisor_N_1) đã được định nghĩa trong hàm process_financial_data
    except Exception as e: 
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.") 

else: 
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
