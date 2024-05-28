import gradio as gr
from logisticRegression import LogisticRegressionModel
from xgBoost import XGBoostModel
from aligner import try_aligner
import errant
# model = LogisticRegressionModel()
model = XGBoostModel()
model.train()
errant_ = errant.load('en')

def anotate_sent(original_sent, corrected_sent):
    anonated = try_aligner.align(errant_, original_sent, corrected_sent) # використовую старий код вирівнювання
    return classify_errors(anonated)

def classify_errors(annotated_sent):
    example_predictions = model.test_example(annotated_sent)
    errors = ""
    for annotation, prediction in zip(annotated_sent.iter_annotations(), example_predictions):
        error_text = annotation.source_text
        target_text = annotation.top_suggestion
        error_with_type = f"Помилка: ({error_text}) Виправлення: ({target_text}) Тип: {prediction}"
        # errors.append(error_with_type)
        errors += error_with_type + ";  \n"
    return errors


with gr.Blocks() as demo:
    with gr.Row():
        original = gr.Textbox(label="Введіть оригінальне речення з граматичними помилками: ")
        corrected = gr.Textbox(label="Введіть виправлене речення: ")
    with gr.Row():
        classify_btn = gr.Button(value="Натисніть, щоб класифікувати помилки")
    with gr.Row():
        classified_sentence = gr.Textbox(label="Класифіковані типи граматичних помилок: ")
    with gr.Row():
        clear_btn = gr.ClearButton(value= "Очистити", components=[original, corrected, classified_sentence])
    with gr.Blocks():
        examples = gr.Examples(
            [["Привіт, Настя:", "Привіт, Насте!"],
             ["Зараз Оленка йшла зі школи.", "Зараз Оленка йде зі школи."]], inputs=[original, corrected], label="Приклади речень")

    classify_btn.click(anotate_sent, inputs=[original, corrected], outputs=classified_sentence)
    clear_btn.click()
demo.launch()

