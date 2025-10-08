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

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Thay vì raise, có thể trả về df và để phần hiển thị báo lỗi.
        # Nhưng để code chạy tiếp, ta sẽ dùng một giá trị mặc định rất nhỏ.
        tong_tai_san_N_1 = 1e-9
        tong_tai_san_N = 1e-9
    else:
        tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
        tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Xử lý giá trị 0 thủ công cho mẫu số.
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Tự động (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
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

# --------------------------------------------------------------------------------------
# 🌟 PHẦN MỚI: CẤU HÌNH VÀ HÀM CHO KHUNG CHAT (CHỨC NĂNG 6) 🌟
# --------------------------------------------------------------------------------------

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Khởi tạo dữ liệu tài chính đã xử lý cho chat
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

# Hàm xử lý gửi tin nhắn chat
def chat_with_gemini(prompt, data_context, api_key):
    """Xử lý tin nhắn chat, gửi nội dung và lịch sử đến Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # Xây dựng lịch sử cho API
        # Lịch sử chat của Streamlit: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        # Lịch sử chat của Gemini: [Content(role='user', parts=[Part.from_text('...')]), ...]
        
        # Thêm ngữ cảnh dữ liệu vào tin nhắn đầu tiên của user
        context_prompt = f"""
        Ngữ cảnh dữ liệu tài chính (Đã xử lý Tốc độ tăng trưởng và Tỷ trọng):
        {data_context}
        
        Đây là câu hỏi của người dùng: {prompt}
        """
        
        # Chuẩn bị lịch sử: Lấy tối đa 5 cặp gần nhất để tránh vượt context
        gemini_history = []
        # Tin nhắn mới nhất của user không cần thêm ngữ cảnh (vì đã thêm vào prompt ở trên)
        if st.session_state.messages:
             # Lấy các tin nhắn cũ, bỏ qua tin nhắn mới nhất
            for message in st.session_state.messages:
                gemini_history.append(
                    genai.types.Content(
                        role="user" if message["role"] == "user" else "model",
                        parts=[genai.types.Part.from_text(message["content"])]
                    )
                )

        
        # Tin nhắn cuối cùng của user (có chứa ngữ cảnh)
        gemini_history.append(
            genai.types.Content(
                role="user", 
                parts=[genai.types.Part.from_text(context_prompt)]
            )
        )
            
        # Gọi API Chat
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_history
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}. Vui lòng kiểm tra Khóa API."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --------------------------------------------------------------------------------------
# 🏃 CHẠY ỨNG DỤNG STREAMLIT 🏃
# --------------------------------------------------------------------------------------

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        
        # Lưu dữ liệu đã xử lý vào session state để dùng cho chat
        st.session_state.df_processed = df_processed.copy()
        
        # Reset lịch sử chat khi file mới được tải lên
        st.session_state.messages = []

        # Hiển thị các chức năng phân tích
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
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Xử lý chia cho 0
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
                thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                 st.warning("Lỗi: Nợ ngắn hạn bằng 0, chỉ số thanh toán hiện hành là vô hạn.")
                 thanh_toan_hien_hanh_N = "Vô hạn"
                 thanh_toan_hien_hanh_N_1 = "Vô hạn"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            # Sử dụng lại logic chuẩn bị dữ liệu cho AI (cả hai chức năng 5 và 6 đều dùng)
            
            # Xác định các giá trị cần thiết
            try:
                tsnh_tang_truong = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]
            except IndexError:
                tsnh_tang_truong = "N/A"
            
            
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
                    f"{thanh_toan_hien_hanh_N_1}" if isinstance(thanh_toan_hien_hanh_N_1, (float, int)) else thanh_toan_hien_hanh_N_1, 
                    f"{thanh_toan_hien_hanh_N}" if isinstance(thanh_toan_hien_hanh_N, (float, int)) else thanh_toan_hien_hanh_N
                ]
            })
            
            data_context_markdown = data_for_ai_df.to_markdown(index=False)
            
            # Chức năng 5
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_context_markdown, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

# --------------------------------------------------------------------------------------
# 💬 PHẦN MỚI: KHUNG CHAT TƯƠNG TÁC (CHỨC NĂNG 6) 💬
# --------------------------------------------------------------------------------------
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
                    st.session_state.messages.pop() # Xóa tin nhắn user nếu lỗi API
                elif st.session_state.df_processed is None:
                     st.error("Lỗi: Không tìm thấy dữ liệu tài chính đã xử lý. Vui lòng tải lại file.")
                     st.session_state.messages.pop()
                else:
                    with st.chat_message("assistant"):
                        with st.spinner("Đang chờ Gemini phản hồi..."):
                            # Chuẩn bị dữ liệu context từ df_processed (tận dụng lại logic của Chức năng 5)
                            # Cần đảm bảo data_context_markdown đã được tạo (nó được tạo ở Chức năng 5 nếu có file)
                            
                            full_response = chat_with_gemini(
                                prompt=prompt, 
                                data_context=data_context_markdown, # Dùng dữ liệu context đã chuẩn bị
                                api_key=api_key
                            )
                            
                            st.markdown(full_response)
                        
                        # Thêm phản hồi của AI vào lịch sử chat
                        st.session_state.messages.append({"role": "assistant", "content": full_response})


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# --------------------------------------------------------------------------------------
