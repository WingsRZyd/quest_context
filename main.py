from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st


@st.cache(allow_output_mutation = True)
def get_pipe(question):
    model_name = "deepset/roberta-base-squad2"

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return nlp


def answer_question(pipe, question, context):

    # a) Get predictions
    QA_input = {
        'question': question,
        'context': context
    }
    res = pipe(QA_input)

    output = res["answer"]
    return output


def main():
    pipe = get_pipe("question")

    st.title("The answer to the question with the available context")

    st.write("This app answers your question based on the context you gave")
    st.write("Model source: https://huggingface.co/deepset/roberta-base-squad2?context=My+name+is+Wolfgang+and+I+live+in+Berlin.+i+working+in+Apple.+but+sometimes+work+in+Micr.&question=Where+do+you+work%3F")

    with st.form("my_form"):
        question = st.text_area("Enter your question here:")
        context = st.text_area("Enter your context here:")

        submitted = st.form_submit_button("Submit")
        # get_answer(submitted, question, context)

        if submitted:
            if question == "" or context == "":
                st.error("Add question and context")

            else:

                result = answer_question(pipe, question, context)

                st.write("**Result:**")
                st.write(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
