"""Build a Word paper, PDF paper, and paper source from measured experiments.

Run after scripts/run_experiments.py. The presentation has a separate JS builder.
The PDF uses the same content blocks as the DOCX and can be rebuilt without Word.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, NextPageTemplate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
TITLE = "VeriFuse Multimodal Content Verification with Transparent Caption Image Consistency Models"
REFS = [
    'K. Nakamura, S. Levy, and W. Y. Wang, "Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection," in Proc. LREC, 2020, pp. 6149-6157. https://aclanthology.org/2020.lrec-1.755/',
    'A. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," in Proc. IEEE/CVF ICCV, 2019. https://arxiv.org/abs/1901.08971',
    'A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. ICML, 2021. https://arxiv.org/abs/2103.00020',
]


def num(value):
    return "n/a" if value is None else f"{value:.3f}"


def load():
    path = ROOT / "results" / "experiments.json"
    if not path.exists():
        raise SystemExit("Run scripts/run_experiments.py before building deliverables.")
    results = json.loads(path.read_text(encoding="utf-8"))
    experiments = {item["id"]: item for item in results["experiments"]}
    required = {"text_baseline", "image_baseline", "feature_fusion", "late_fusion", "modality_ablation", "ood_generalization"}
    if not required.issubset(experiments):
        raise ValueError(f"Missing experiment IDs: {required - experiments.keys()}")
    return results, experiments


def compact(obj):
    return json.dumps(obj, ensure_ascii=True, separators=(", ", ": "))


def variant_rows(experiment):
    variants = experiment.get("variants", [])
    if isinstance(variants, dict):
        variants = [{"name": key, **(value if "metrics" in value else {"metrics": value})} for key, value in variants.items()]
    result = []
    for item in variants:
        m = item.get("metrics", item)
        result.append([item.get("name", item.get("id", item.get("model", "Variant"))).replace("_", " "), num(m.get("accuracy")), num(m.get("f1")), num(m.get("roc_auc"))])
    return result


def content(results, exp):
    fusion = exp["feature_fusion"]["metrics"]
    text = exp["text_baseline"]["metrics"]
    image = exp["image_baseline"]["metrics"]
    late = exp["late_fusion"]["metrics"]
    ood = exp["ood_generalization"]["metrics"]
    blocks = []
    def p(text): blocks.append(("p", text))
    def h(text): blocks.append(("h", text))
    def t(caption, headers, rows, widths=None): blocks.append(("table", (caption, headers, rows, widths)))
    p(f"Abstract - Multimodal misinformation can contain a plausible caption and a plausible image whose meanings disagree. This course project studies that relation in a reproducible controlled setting. VeriFuse contains 864 balanced synthetic image-caption pairs derived from 432 source scenes. Grouped splits prevent an original scene and its manipulated variant from crossing evaluation boundaries. We compare train-only TF-IDF and pixel-attribute logistic baselines, feature fusion with explicit cross-modal interactions, and validation-selected late fusion. On the 144-pair in-distribution test set, feature fusion achieves accuracy {num(fusion['accuracy'])} and F1 {num(fusion['f1'])}, compared with F1 {num(text['f1'])} for text and {num(image['f1'])} for image alone. A separate 144-pair count-manipulation test produces F1 {num(ood['f1'])}. Modality ablations, 16 difficult cases, grounded feature contributions, and perturbation checks expose what the model uses and where its scope ends. These measurements concern constructed scene-caption consistency. They do not establish natural-image deepfake detection, news truth, or real-world calibration.")
    p("Index Terms - multimodal NLP, image caption consistency, feature fusion, explainability, controlled benchmark.")
    h("I Introduction")
    p("The CSET 346 project asks whether combining language and visual evidence improves detection and supports useful explanations. A caption may describe the wrong object even when neither its language nor its image looks implausible in isolation. This relational problem motivates a joint model: the label depends on agreement across modalities, so a classifier receiving only one modality may lack the necessary information.")
    p("VeriFuse implements the text-plus-image option as a small, inspectable research prototype. The task predicts REAL for an unaltered procedural scene paired with a correct caption and FAKE when one attribute in the image or caption deliberately contradicts the other. The interactive system also returns UNCERTAIN for unsupported inputs. The label vocabulary follows the course brief, but REAL does not certify an actual event and FAKE does not imply deceptive intent.")
    p("The contribution is an end-to-end reproducible experiment with two single-modality baselines, two fusion strategies, a withheld manipulation type, missing-modality tests, and explanations tied to model features. The experiment deliberately uses a simple domain to make each source of evidence measurable. Its high in-domain scores, when present, must be read in that context.")
    h("II Background and Task Boundary")
    p("Fakeddit provides a multimodal benchmark for fine-grained fake-news research [1]. FaceForensics++ studies manipulated facial imagery [2]. These datasets illustrate distinct evaluation targets: caption-context agreement and image manipulation artifacts need not be the same problem. CLIP learns transferable visual representations using natural-language supervision [3], making pretrained vision-language encoders a plausible future comparison. VeriFuse does not download or train on these datasets and does not use CLIP or another pretrained encoder. The citations provide background rather than benchmark equivalence.")
    p("Our chosen scope is controlled cross-modal contradiction, which the course brief permits through carefully constructed data and misleading captions. We exclude faces, identity judgments, lip synchronization, audio, source authentication, and external factual retrieval. A system that agrees with a caption can still be wrong about the world. This distinction guides both the UI wording and interpretation of the metrics.")
    h("III Dataset and Preprocessing")
    p("The generator creates 256 by 256 pixel RGB scenes using four colors (red, blue, green, yellow), three shapes (circle, square, triangle), and counts of one to three objects. All 36 combinations recur within each split. Small deterministic changes in position, size, foreground color, and background color create source scenes. Rendering at three times the final resolution followed by downsampling produces anti-aliased boundaries. Four caption templates vary surface wording without changing the intended attributes.")
    p("Each source group contains a matching REAL pair and one FAKE pair. The FAKE record changes exactly one attribute in either the caption or the image. The in-distribution splits use color and shape contradictions; only the held-out generalization split uses count contradictions. Thus the label records a controlled intervention, rather than a claim inferred from an annotator or filename. The image model reads image pixels and does not receive manifest labels or scene metadata.")
    t("TABLE I Dataset split by source scene", ["Split", "Groups", "Pairs", "REAL", "FAKE"], [["Train",216,432,216,216],["Validation",72,144,72,72],["Test",72,144,72,72],["Count OOD",72,144,72,72],["Total",432,864,432,432]], [1.12,.53,.53,.53,.53])
    p(f"The random seed is {results.get('seed', 7031)}. All variants of a source scene remain in one split. This is a scene-group split, not a split of random rows. It prevents an original image and its altered partner from directly leaking across boundaries, although all splits still share the same generator and visual grammar. The corpus is procedural original content and contains no collected personal media. Manifest JSON and CSV files preserve identifiers, group, split, source, image path, caption, label, and manipulation metadata.")
    p("Text preprocessing lowercases and tokenizes words and numbers. A train-only vocabulary includes unigrams and adjacent bigrams appearing in at least two training documents. Sublinear term frequency and smoothed inverse document frequency produce L2-normalized vectors. Caption parsing separately resolves supported color, shape and count words and a small explicit synonym set. Ambiguous or negated captions receive special handling during interactive inference.")
    p("Image preprocessing resizes to 128 by 128 pixels. A saturation mask isolates colored foreground components. Four-neighbor connected components above the minimum area estimate object count and bounding regions. Mean foreground RGB chooses the nearest supported color. Component fill ratio separates squares, circles and triangles. The 15 image features contain one-hot attributes plus foreground fraction, mean RGB and component fill. This hand-designed representation assumes isolated, similarly colored shapes on a pale background.")
    h("IV Models and Explainable Verification")
    p("Each learned classifier is a logistic model implemented with NumPy. Training standardizes each feature using training means and scales, optimizes binary cross-entropy with L2 regularization 0.03 through 1400 full-batch Adam steps at learning rate 0.07, and leaves the intercept unpenalized. Predictions use p(FAKE) = sigmoid(w^T z + b). The fitted vocabulary has 93 text features; feature fusion has 114 total features. The saved bundle includes weights, normalizers, vocabulary, configuration and the selected fusion setting. It permits local inference without retraining.")
    p("The text baseline learns from TF-IDF alone. The image baseline learns from the 15 pixel-derived features. Feature fusion concatenates both representations with six explicit interaction features: color contradiction, shape contradiction, count contradiction, contradiction fraction, all-attributes match, and known-attribute fraction. These interactions encode the central relation the single-modality models cannot directly observe. Because domain knowledge enters through these features, results do not isolate a benefit of representation concatenation alone.")
    p("Late fusion combines separately trained probabilities as alpha times the text score plus (1 - alpha) times the image score. Validation log loss selects alpha = 0.0 in this run, so the reported late-fusion result equals the image baseline. This degenerate optimum is retained rather than replaced after observing test performance. A 0.5 threshold defines binary experiment predictions. No test labels enter vocabulary fitting, feature normalization, model fitting or validation selection.")
    p("The explanation uses local probability perturbations. Deleting a caption token and rescoring gives its signed effect. Painting each detected object's rectangle with the light background and rescoring gives a region effect. Holding encoders fixed while flipping one attribute comparison measures its counterfactual effect on p(FAKE). The system shows relevant words and image regions and gives an attribute-based evidence statement. This is evidence grounding within the extractor, not a learned saliency map. Unsupported scenes or captions, or confidence below 0.58, yield UNCERTAIN. The score has no demonstrated real-world probability calibration.")
    h("V Experimental Protocol")
    p("The six requested experiment groups are: (1) compare both single-modality models with feature fusion; (2) compare feature and late fusion; (3) test generalization to count manipulation withheld from training; (4) remove text or image evidence at inference; (5) inspect at least 15 difficult predictions; and (6) test explanation relevance through perturbation and faithful contribution accounting. The results JSON stores all measurements, confusion matrices, class-wise scores and threshold curves. The numbered requirements map differs from individual model IDs because several required comparisons reuse the same run.")
    p("We report accuracy, precision, recall, F1, ROC-AUC and PR-AUC with FAKE as the positive class. PR-AUC is average precision, calculated by the step integral of precision against recall. Tied scores share a threshold. Confusion matrices use rows [REAL, FAKE] and columns [REAL, FAKE]. Class-wise scores accompany the aggregate comparison. Results come from one deterministic seed and one split, so no statistical significance or sampling confidence interval is claimed.")
    h("VI Results")
    rows=[]
    for identifier, label in [("text_baseline","Text"),("image_baseline","Image"),("feature_fusion","Feature"),("late_fusion","Late")]:
        m=exp[identifier]["metrics"]
        rows.append([label]+[num(m[k]) for k in ("accuracy","precision","recall","f1","roc_auc","pr_auc")])
    t("TABLE II In distribution test metrics", ["Model","Acc","Prec","Rec","F1","ROC","AP"], rows, [.67,.45,.45,.45,.45,.45,.45])
    p(f"Feature fusion achieves test F1 {num(fusion['f1'])}; text-only F1 is {num(text['f1'])}, image-only F1 is {num(image['f1'])}, and late-fusion F1 is {num(late['f1'])}. These measured differences support using explicit agreement information on this task. Weak unimodal performance is expected because the generator keeps individual scenes and captions plausible. It should not be described as evidence that unimodal classifiers are generally ineffective for deepfakes.")
    cm=fusion["confusion_matrix"]
    t("TABLE III Feature fusion confusion matrix", ["Actual", "Pred REAL", "Pred FAKE"], [["REAL",cm[0][0],cm[0][1]],["FAKE",cm[1][0],cm[1][1]]], [1.14,1.14,1.14])
    t("TABLE IV Feature fusion class wise metrics", ["Class","Precision","Recall","F1","n"], [[label,num(m['precision']),num(m['recall']),num(m['f1']),m['support']] for label,m in fusion['per_class'].items()], [.65,.78,.7,.7,.5])
    h("A Generalization and ablation")
    p(f"The withheld count-manipulation split gives feature-fusion accuracy {num(ood['accuracy'])} and F1 {num(ood['f1'])}. Count contradiction is unseen during fitting, but the architecture contains a general contradiction fraction and all-attributes-match feature. Therefore success here indicates transfer through a designed consistency rule within the same scene domain; it does not establish cross-dataset or natural-image generalization.")
    vr=variant_rows(exp["ood_generalization"])
    if vr: t("TABLE V Withheld count manipulation",["Model","Acc","F1","ROC"],vr,[1.54,.63,.63,.63])
    vr=variant_rows(exp["modality_ablation"])
    if vr: t("TABLE VI Modality ablation",["Condition","Acc","F1","ROC"],vr,[1.54,.63,.63,.63])
    p("The ablation measures a fixed trained model with evidence removed, so it tests dependence at inference rather than the best model retrained for missing data. Missing inputs change the feature distribution. The interactive application consequently treats unavailable or unsupported evidence conservatively. Binary ablation metrics and selective UNCERTAIN decisions answer different questions and should not be combined into one accuracy score.")
    h("B Explanation quality and sanity checks")
    faith=results.get("faithfulness",{})
    p("The explanation evaluation uses the first 48 fixed test records. Removing attribute tokens and filler tokens measures local sensitivity. Repairing a false caption and introducing a color conflict into a matching caption tests whether the score changes in the expected direction. This supports model relevance within the synthetic domain, but does not prove human explanation usefulness or forged-region localization.")
    t("TABLE VII Explanation perturbation results",["Measurement","Value"],[["Mean absolute attribute token effect",num(faith['mean_absolute_attribute_token_effect'])],["Mean absolute filler token effect",num(faith['mean_absolute_filler_token_effect'])],["Mean FAKE score drop after correction",num(faith['mean_fake_probability_drop_after_correction'])],["Correction direction pass rate",num(faith['correction_direction_pass_rate'])],["Mean FAKE score rise after conflict",num(faith['mean_fake_probability_rise_after_conflict'])],["Conflict direction pass rate",num(faith['conflict_direction_pass_rate'])]],[2.6,.83])
    checks=results.get("sanity_checks",[])
    if checks:
        p("The accompanying machine-readable report records sanity checks for genuine pairs, deliberately conflicting pairs, and unsupported or incomplete inputs. Each check retains its expected behavior and observed outcome, so unsupported inputs cannot silently serve as successful REAL predictions.")
    h("VII Difficult Case Analysis")
    p("The saved analysis selects the eight smallest-margin test pairs from each class after fitting, for 16 examples. All are classified correctly and remain confident (the weakest FAKE score is about 0.943), so these are relatively difficult examples, not observed errors. They repeatedly expose color or shape disagreement and matching controls, with several captions describing a single object. Their agreement with the extractor shows the simplicity of the domain rather than robustness to natural language. The report preserves each caption, image path, probability, decision and explanation.")
    cases=results["difficult_cases"]
    assert len(cases)>=15, "At least 15 cases are required"
    for index,case in enumerate(cases,1):
        prob=case.get('fake_probability',case.get('probability'))
        correct=case.get('correct',case.get('prediction')==case.get('label'))
        p(f"Case {index} ({case['id']}). Caption: {case['text']} Ground truth {case['label']}; prediction {case['prediction']}; p(FAKE) {num(prob)}; {'correct' if correct else 'error'}. Manipulation: {case.get('manipulation','unspecified')}. {case.get('explanation','See the saved case evidence in the experiment report.')}")
    h("VIII Limitations and Responsible Use")
    p("The narrow generator favors interpretable agreement rules and cannot represent natural-photo complexity. The hand-designed parser and color/shape extractor already contain much of the task structure. The same rendering process appears in every split, the lexical variation is limited, and the single seed gives no measure of variability across different sampled corpora. The reference models are lightweight educational baselines rather than strong pretrained architectures. High synthetic performance cannot justify state-of-the-art or deployment claims.")
    p("The uncertainty gate is a conservative heuristic, not a calibrated selective-classification study. Domain heuristics can accept an unusual image or reject a valid stylized one. Negation, multiple object categories, unfamiliar words and natural backgrounds need more capable models. Perturbation effects explain local score changes, yet do not establish causal truth. Token deletion and region masking can also move an input outside the training distribution.")
    p("The generated dataset avoids collecting private individuals' media and does not depict real people. An extension to public data must document licenses, consent requirements, provenance, annotator disagreement and demographic or source bias. A deployment must separately assess false accusations and false reassurance. VeriFuse should support coursework and experimentation; it should not make moderation decisions or accuse a person of deception.")
    h("IX Conclusion and Reproducibility")
    p(f"VeriFuse provides a local multimodal verification experiment with measured baselines, two fusion approaches, held-out manipulation testing, ablations and grounded explanations. Feature fusion reaches F1 {num(fusion['f1'])} on the controlled test and {num(ood['f1'])} on withheld count manipulations. The result is evidence for explicit cross-modal interactions within this generator. Broader claims require licensed natural data, stronger encoders, source-separated evaluation, repeated seeds, calibration and human review of explanations.")
    p("To reproduce the delivered run, install the listed Python dependencies, generate the dataset, and run the experiment script from the repository root. The project README gives exact commands. Source data live in data/manifest.json, model parameters in models/bundle.json, and measured evidence in results/experiments.json. Paper and slide builders consume that results file. The author names, student identifiers, affiliation details and repository URL should be added by the submitting team before submission; none are fabricated here.")
    h("References")
    for index,ref in enumerate(REFS,1): p(f"[{index}] {ref}")
    return blocks


def build_docx(blocks):
    doc=Document()
    sec=doc.sections[0]
    sec.page_width,sec.page_height=Inches(8.27),Inches(11.69)
    sec.top_margin=sec.bottom_margin=Inches(.65)
    sec.left_margin=sec.right_margin=Inches(.65)
    style=doc.styles['Normal']; style.font.name='Times New Roman'; style.font.size=Pt(10)
    style.paragraph_format.space_after=Pt(4)
    style.paragraph_format.line_spacing=1
    for name in ('Title','Subtitle','Heading 1','Heading 2'):
        doc.styles[name].font.name='Times New Roman'; doc.styles[name].font.color.rgb=RGBColor(0,0,0)
    title=doc.add_paragraph(TITLE,'Title'); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    title.style.font.size=Pt(18)
    sub=doc.add_paragraph('CSET 346 course project  |  Bennett University','Subtitle');sub.alignment=WD_ALIGN_PARAGRAPH.CENTER;sub.style.font.size=Pt(11)
    p=doc.add_paragraph('Student names and IDs: [complete before submission]');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    sec=doc.add_section(WD_SECTION_START.CONTINUOUS)
    cols=sec._sectPr.xpath('./w:cols')[0];cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'288')
    for kind,value in blocks:
        if kind=='h':
            para=doc.add_paragraph(value,'Heading 1');para.style.font.size=Pt(10);para.style.font.bold=True
            para.paragraph_format.space_before=Pt(9);para.paragraph_format.space_after=Pt(4)
        elif kind=='p':
            para=doc.add_paragraph(value);para.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        else:
            caption,headers,rows,widths=value
            para=doc.add_paragraph(caption);para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.paragraph_format.keep_with_next=True
            for run in para.runs:run.font.size=Pt(8)
            table=doc.add_table(rows=1,cols=len(headers));table.autofit=False
            for row_index,values in enumerate([headers]+rows):
                cells=table.rows[0].cells if row_index==0 else table.add_row().cells
                for j,(cell,text) in enumerate(zip(cells,values)):
                    cell.text=str(text)
                    if widths:cell.width=Inches(widths[j])
                    tcPr=cell._tc.get_or_add_tcPr()
                    borders=OxmlElement('w:tcBorders')
                    for edge in ('top','left','bottom','right'):
                        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
                    tcPr.append(borders)
                    if row_index==0:
                        fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'E8ECF0');tcPr.append(fill)
                    for para in cell.paragraphs:
                        para.paragraph_format.space_after=Pt(3);para.paragraph_format.space_before=Pt(3)
                        para.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                        for run in para.runs:run.font.size=Pt(7.5);run.font.bold=row_index==0
                if row_index==0:
                    repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
    footer=doc.sections[-1].footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('VeriFuse course project  |  ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    for run in footer.runs:run.font.size=Pt(8)
    doc.core_properties.title=TITLE;doc.core_properties.subject='CSET 346 multimodal NLP research prototype';doc.core_properties.author='CSET 346 course project'
    doc.save(OUT/'VeriFuse_Paper.docx')


def build_pdf(blocks):
    folder=OUT/'pdf';folder.mkdir(parents=True,exist_ok=True)
    width,height=8.27*inch,11.69*inch
    margin=.65*inch;gap=.2*inch;column=(width-2*margin-gap)/2
    doc=BaseDocTemplate(str(folder/'VeriFuse_Paper.pdf'),pagesize=(width,height),leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=margin,title=TITLE,author='CSET 346 course project')
    def footer(canvas,document):
        canvas.setFont('Times-Roman',8);canvas.setFillColor(colors.HexColor('#60666d'))
        canvas.drawCentredString(width/2,26,f'VeriFuse course project  |  {document.page}')
    normal_frames=[Frame(margin,margin,column,height-2*margin,id='left'),Frame(margin+column+gap,margin,column,height-2*margin,id='right')]
    first_frames=[Frame(margin,height-margin-110,width-2*margin,110,id='title'),Frame(margin,margin,column,height-2*margin-115,id='left-first'),Frame(margin+column+gap,margin,column,height-2*margin-115,id='right-first')]
    doc.addPageTemplates([PageTemplate(id='First',frames=first_frames,onPage=footer,autoNextPageTemplate='Body'),PageTemplate(id='Body',frames=normal_frames,onPage=footer)])
    body=ParagraphStyle('Body',fontName='Times-Roman',fontSize=9.5,leading=11,spaceAfter=5,alignment=TA_JUSTIFY)
    heading=ParagraphStyle('Heading',parent=body,fontName='Times-Bold',fontSize=10,leading=12,spaceBefore=9,spaceAfter=5,keepWithNext=True,alignment=0)
    cap=ParagraphStyle('Caption',parent=body,fontSize=8,leading=9,alignment=TA_CENTER,keepWithNext=True,spaceBefore=6)
    cellstyle=ParagraphStyle('Cell',parent=body,fontSize=7.2,leading=9,spaceAfter=0,alignment=TA_CENTER)
    story=[Paragraph(escape(TITLE),ParagraphStyle('Title',fontName='Times-Roman',fontSize=18,leading=21,alignment=TA_CENTER,spaceAfter=9)),Paragraph('CSET 346 course project<br/>Bennett University<br/>Student names and IDs: [complete before submission]',ParagraphStyle('Subtitle',fontName='Times-Roman',fontSize=10,leading=13,alignment=TA_CENTER))]
    from reportlab.platypus import FrameBreak
    story.append(FrameBreak())
    for kind,value in blocks:
        if kind=='h':story.append(Paragraph(escape(value),heading))
        elif kind=='p':story.append(Paragraph(escape(value),body))
        else:
            caption,headers,rows,widths=value
            story.append(Paragraph(escape(caption),cap))
            data=[[Paragraph(escape(str(v)),cellstyle) for v in row] for row in [headers]+rows]
            sizes=[column*(v/sum(widths)) for v in widths] if widths else None
            table=Table(data,colWidths=sizes,repeatRows=1,hAlign='CENTER')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8ECF0')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
            story.extend([table,Spacer(1,4)])
    doc.build(story)


def build_markdown(blocks):
    parts=['# '+TITLE,'','CSET 346 course project, Bennett University','Student names and IDs: [complete before submission]','']
    for kind,value in blocks:
        if kind=='h':parts+=['## '+value,'']
        elif kind=='p':parts+=[value,'']
        else:
            caption,headers,rows,_=value
            parts += [caption,'','| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
            parts += ['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
    (ROOT/'docs'/'paper.md').write_text('\n'.join(parts),encoding='utf-8')
    (ROOT/'docs'/'references.bib').write_text('''@inproceedings{nakamura2020fakeddit,
  title={Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection},
  author={Nakamura, Kai and Levy, Sharon and Wang, William Yang},
  booktitle={Proceedings of LREC}, pages={6149--6157}, year={2020},
  url={https://aclanthology.org/2020.lrec-1.755/}
}
@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rossler, Andreas and others}, booktitle={IEEE/CVF ICCV}, year={2019},
  url={https://arxiv.org/abs/1901.08971}
}
@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others}, booktitle={ICML}, year={2021},
  url={https://arxiv.org/abs/2103.00020}
}
''',encoding='utf-8')


if __name__=='__main__':
    results,experiments=load();OUT.mkdir(parents=True,exist_ok=True)
    blocks=content(results,experiments)
    build_docx(blocks);build_pdf(blocks);build_markdown(blocks)
    print('Created Word paper, PDF paper, Markdown source and BibTeX references from measured results.')
