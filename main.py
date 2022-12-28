from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st



def answer_question(question, context):
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output = res["answer"]
    return output



def main():
    st.title("The answer to the question with the available context")

    st.write("This app answers your question based on the context you gave")
    st.write("Model source: https://huggingface.co/deepset/roberta-base-squad2?context=My+name+is+Wolfgang+and+I+live+in+Berlin.+i+working+in+Apple.+but+sometimes+work+in+Micr.&question=Where+do+you+work%3F")

    with st.form("my_form"):
        question = st.text_area("Enter your question here:")
        context = st.text_area("Enter your context here:")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if question == "" or context == "":
                st.error("Add question and context")

            else:

                result = answer_question(question, context)

                st.write("**Result:**")
                st.write(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
