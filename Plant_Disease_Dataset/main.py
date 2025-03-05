import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.header('Plant Disease Recognition System')
    image_path = "plant_disease_image.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown('''
    ### Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    **How It Works:**
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    **Why Choose Us?**
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    **Get Started**
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    **About Us**
    Learn more about the project, our team, and our goals on the **About** page.
    ''')

# About Page
elif app_mode == "About":
    st.header("About")
    image_path = "plant_disease_image1.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    ## **About Dataset**
    This dataset is created using offline augmentation from the original dataset. The original dataset can be found on GitHub. The dataset consists of around **87,000 RGB images** of healthy and diseased crop leaves categorized into **38 different classes**.

    ### **Dataset Structure**
    - **Train Set**: 70,295 images  
    - **Validation Set**: 17,572 images  
    - **Test Set**: 33 images  

    ---
    
    ## **Future Goals**
    
    We are committed to developing an advanced application that leverages technology to enhance plant health management. Our key objectives are:
    
    ### **1. Early Disease Prediction**
    - The application predicts plant diseases **before they occur** by analyzing plant scans and environmental data.
    - Helps farmers take **proactive measures**, reducing crop loss and increasing yield.
    
    ### **2. Automated Health Reports**
    - Generates **detailed health reports every 15 days** after scanning the plant.
    - Provides insights into **plant conditions** to track changes over time.
    
    ### **3. Environmental Data Integration**
    - Considers crucial factors such as **temperature, humidity, and natural conditions**.
    - Ensures **accurate disease prediction** and treatment recommendations.
    
    ### **4. Fertilizer Recommendations**
    - Suggests **suitable fertilizers** based on the detected disease and climate conditions.
    - Helps optimize **plant nutrition**, ensuring healthy growth and improved productivity.
    
    By integrating **technology with agriculture**, our goal is to empower farmers with **data-driven insights**, leading to **sustainable and efficient farming practices**. 🚀🌱
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    
    # Predict Button
    if st.button("Predict"):
        with st.spinner("Please Wait..."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Define Class
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
