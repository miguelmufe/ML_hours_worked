from reportlab.pdfgen import canvas

def convert_to_pdf_reportlab(input_file, output_pdf):
    pdf_canvas = canvas.Canvas(output_pdf)

    with open(input_file, 'r') as python_file:
        content = python_file.read()
        pdf_canvas.setPageSize((595.27, 841.89))  # Set to A4 size
        text_object = pdf_canvas.beginText(40, 800)
        text_object.setFont("Helvetica", 10)

        for line in content.splitlines():
            text_object.textLine(line)
            if text_object.getY() < 40:  # Check if the text reaches the bottom of the page
                pdf_canvas.drawText(text_object)
                pdf_canvas.showPage()
                text_object = pdf_canvas.beginText(40, 800)
                text_object.setFont("Helvetica", 10)

        pdf_canvas.drawText(text_object)
    pdf_canvas.save()

for file in ['main.py', 'Preprocessing.py', 'Guess.py', 'predictions.py',
             'Random_forest.py', 'Linear_SGD.py', 'Regression_trees.py', 'KNN.py']:
    convert_to_pdf_reportlab(file, f'{file[:-3]}.pdf')