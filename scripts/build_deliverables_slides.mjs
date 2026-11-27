// Editable academic slides from the actual VeriFuse experiment output.
// Requires the Codex bundled @oai/artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const runtime = process.env.CODEX_ARTIFACT_RUNTIME || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill = process.env.CODEX_PRESENTATION_SKILL || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.921.10847/skills/presentations');
process.env.RUNTIME_NODE_MODULES ||= path.join(runtime,'node/node_modules');
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data = JSON.parse(await fs.readFile(path.join(root,'results/experiments.json'),'utf8'));
const ex = Object.fromEntries(data.experiments.map(x=>[x.id,x]));
const out = path.join(root,'output');
const build = path.join(out,'.build','slides');
await fs.mkdir(build,{recursive:true});
const qa = path.join(root,'scripts','build_deliverables_qa');
await fs.mkdir(qa,{recursive:true});
const family = resolvePresentationFont({fontFamily:'Arial'});
const p = Presentation.create({slideSize:{width:1280,height:720}});
const C = {navy:'#11283F',ink:'#172B3E',muted:'#52677B',teal:'#147C85',cream:'#F6F5EF',grid:'#D5DDE4',paper:'#FFFFFF'};
const n=x=>Number(x).toFixed(3);
const refs = '[1] K. Nakamura, S. Levy, and W. Y. Wang, Fakeddit, LREC 2020, pp. 6149–6157. https://aclanthology.org/2020.lrec-1.755/\n[2] A. Rossler et al., FaceForensics++, ICCV 2019. https://arxiv.org/abs/1901.08971\n[3] A. Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021. https://arxiv.org/abs/2103.00020';
function txt(slide,text,x,y,w,h,size=28,color=C.ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=text;shape.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return shape;
}
function slide(title,notes=''){
  const s=p.slides.add();s.background.fill=C.paper;
  txt(s,title,64,42,1152,94,44,C.navy,true);
  txt(s,String(p.slides.items.length).padStart(2,'0'),1160,666,60,25,16,C.muted);
  s.speakerNotes.textFrame.setText(notes+'\nSource: results/experiments.json, seed '+data.seed+'. Synthetic scene-caption consistency only.');
  return s;
}
function table(s,headers,rows,x,y,w,h,colWidths){
  const t=s.tables.add({rows:rows.length+1,columns:headers.length,left:x,top:y,width:w,height:h,values:[headers,...rows],columnWidths:colWidths});
  t.borders.assign({style:'solid',fill:C.grid,width:1});
  for(let r=0;r<=rows.length;r++) for(let c=0;c<headers.length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2?C.paper:'#F0F4F6');
    cell.text.style={typeface:family,fontSize:22,color:r===0?C.paper:C.ink,bold:r===0};
  }
  return t;
}
function chart(s,categories,series,x,y,w,h){
  // Charts display three decimals; store that precision for portable Excel workbooks.
  series=series.map(item=>({...item,values:item.values.map(value=>Number(value.toFixed(3)))}));
  const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:85},hasLegend:series.length>1,legend:{position:'bottom',textStyle:{typeface:family,fontSize:18}},xAxis:{textStyle:{typeface:family,fontSize:20,fill:C.ink},majorGridlines:null},yAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0.00',textStyle:{typeface:family,fontSize:18,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:family,fontSize:20,fill:C.ink}},chartFill:'none',plotAreaFill:'none'});
  applyPresentationChartFont(c,{fontFamily:family});return c;
}
async function picture(s,relative,x,y,w,h,alt){s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,relative))),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}

// 1 Cover. All foreground content remains editable.
{
 const s=p.slides.add();s.background.fill=C.navy;
 txt(s,'VeriFuse',72,90,1110,110,82,C.paper,true);
 txt(s,'Multimodal content verification',76,217,1100,66,42,C.paper);
 txt(s,'A controlled study of caption and image consistency',77,302,1080,72,29,'#BBD8DE');
 txt(s,'CSET 346 Natural Language Processing\nBennett University',78,480,960,70,25,C.paper);
 txt(s,'Student names and IDs: [complete before submission]',78,607,1080,35,22,'#D7E0E8');
 s.speakerNotes.textFrame.setText('Course project from PJ_07. This is an educational prototype. REAL means a matching procedural scene-caption pair. FAKE means a deliberate attribute contradiction. It is not evaluated as a facial deepfake or news-truth detector.');
}
// 2 Definition.
{
 const s=slide('Verification task and scope','Research question: does fusion improve on single-modality models, and can the result cite relevant evidence? '+refs);
 txt(s,'Does a short caption agree with the visible scene?',66,157,1120,70,34,C.teal,true);
 table(s,['Output','Meaning in this project'],[['REAL','Original synthetic scene and matching caption'],['FAKE','One altered attribute creates a contradiction'],['UNCERTAIN','Unsupported input or confidence below 0.58']],66,265,1148,236,[240,908]);
 txt(s,'The label describes constructed consistency. It does not verify real-world events, people or news.',67,548,1110,90,27,C.muted);
}
// 3 Corpus.
{
 const s=slide('864 pairs from 432 source scenes','Each source group yields an original REAL pair and an altered FAKE pair. Color and shape changes occur in train, validation and test. Count changes appear only in OOD. All scene variants stay within one split. Data source: procedural original content.');
 await picture(s,'data/images/scene-0000.png',65,165,365,365,'One generated training scene, used only to illustrate the corpus');
 txt(s,'Procedural scene example\n4 colors, 3 shapes, 3 counts',65,548,390,82,24,C.muted);
 table(s,['Split','Pairs','REAL','FAKE'],[['Train',432,216,216],['Validation',144,72,72],['Test',144,72,72],['Count OOD',144,72,72]],476,178,737,321,[285,150,151,151]);
 txt(s,'Grouped splits keep every original and its altered partner together.',477,549,723,74,26,C.ink);
}
// 4 Methods.
{
 const s=slide('Representations and model fitting','Implementation: verifuse/features.py and verifuse/learning.py. The handcrafted scene extractor assumes isolated same-color shapes on a pale background. All normalizers and the vocabulary fit only on train.');
 table(s,['Component','Representation'],[['Caption','93 TF-IDF unigram and bigram features'],['Image','15 pixel-derived color, shape and count features'],['Joint evidence','6 explicit agreement and contradiction features'],['Classifier','L2 logistic regression with full-batch Adam']],65,163,1148,303,[255,893]);
 txt(s,'Training uses 1,400 steps, learning rate 0.07 and L2 weight 0.03. Test classification uses threshold 0.5.',67,514,1127,92,28,C.ink);
 txt(s,'No pretrained encoder or external benchmark training data',67,625,1115,34,23,C.muted);
}
// 5 Fusion.
{
 const s=slide('Feature fusion and decision fusion','Late fusion alpha grid is selected only by validation log loss. In this run alpha is 0.0, so late fusion exactly matches the image-only baseline. This is an observed outcome, not a test-selected correction.');
 txt(s,'Feature fusion',67,165,1075,47,33,C.teal,true);
 txt(s,'One classifier sees caption features, image features and explicit attribute contradictions. The model has 114 inputs.',67,231,1138,90,29);
 txt(s,'Decision fusion',67,373,1075,47,33,C.teal,true);
 txt(s,'A validation-selected blend combines text and image probabilities. The selected text weight is 0.0 in this run.',67,439,1138,90,29);
 txt(s,'Only feature fusion can directly use the engineered cross-modal interactions.',67,581,1120,62,27,C.muted);
}
// 6 Results.
{
 const s=slide('Explicit interactions solve the controlled test','144 test pairs, 72 per class. FAKE is positive. PR-AUC means average precision. Full class-wise metrics and curves are saved in results/experiments.json. Strong results are expected given the explicit comparisons and narrow generator.');
 const ids=['text_baseline','image_baseline','feature_fusion','late_fusion'];
 chart(s,['Text','Image','Feature','Late'],[{name:'F1',values:ids.map(id=>ex[id].metrics.f1),fill:C.teal,valuesFormatCode:'0.000'}],59,165,602,387);
 table(s,['Model','Acc','F1','ROC'],ids.map((id,i)=>[['Text','Image','Feature','Late'][i],n(ex[id].metrics.accuracy),n(ex[id].metrics.f1),n(ex[id].metrics.roc_auc)]),696,190,518,304,[194,108,108,108]);
 txt(s,'F1 score',68,555,570,37,22,C.muted);
 txt(s,'The evidence concerns synthetic consistency. Perfect scores do not establish real-photo deepfake detection.',67,615,1118,46,24,C.muted);
}
// 7 Confusion.
{
 const m=ex.feature_fusion.metrics,cm=m.confusion_matrix;
 const s=slide('Feature fusion test errors','Rows are actual labels and columns are predictions. The 144-pair test is balanced. Confidence intervals and repeated-seed estimates have not been computed.');
 table(s,['Actual','Pred REAL','Pred FAKE'],[['REAL',cm[0][0],cm[0][1]],['FAKE',cm[1][0],cm[1][1]]],65,200,701,230,[253,224,224]);
 txt(s,'144 / 144',836,215,365,95,58,C.teal,true);
 txt(s,'correct test pairs',838,315,352,65,27,C.ink);
 txt(s,'Both classes have precision, recall and F1 of 1.000 in this run.',67,506,1115,75,29);
 txt(s,'A single seed and one rendering grammar limit the conclusion.',67,615,1110,39,25,C.muted);
}
// 8 OOD and ablation.
{
 const s=slide('Generalization and missing evidence','OOD is a withheld manipulation type within the same generator. Count contradictions are unseen during training but the shared contradiction fraction and all-attributes-match features already encode a general agreement prior. Ablations replace missing feature families by their training means without retraining.');
 txt(s,'Held-out count manipulation',67,158,550,60,28,C.teal,true);
 txt(s,'Inference ablation',704,158,500,60,28,C.teal,true);
 table(s,['Model','F1'],ex.ood_generalization.variants.map(v=>[v.name.replaceAll('_',' '),n(v.metrics.f1)]),65,229,538,287,[366,172]);
 table(s,['Evidence removed','F1'],ex.modality_ablation.variants.map(v=>[v.name.replace('Without ',''),n(v.metrics.f1)]),703,229,508,233,[352,156]);
 txt(s,'The designed consistency features transfer to count changes. Removing joint evidence reduces F1 to 0.500.',67,573,1129,91,28,C.muted);
}
// 9 Explanation sample, original image evidence.
{
 const c=data.difficult_cases[0];
 const s=slide('Grounded evidence for a color conflict','Case '+c.id+'. '+c.explanation+' Image and caption are source evidence from the generated test set. Counterfactual effects flip one comparison while keeping encoders fixed.');
 await picture(s,c.image_path,64,161,399,399,'Held-out image showing the actual colored shape in the selected contradiction case');
 txt(s,'Caption',520,167,680,43,24,C.muted);
 txt(s,c.text,519,217,684,114,34,C.ink,true);
 txt(s,'Caption color: yellow\nExtracted image color: green',521,357,680,98,28,C.ink);
 txt(s,'FAKE probability  '+n(c.fake_probability),521,501,680,63,33,C.teal,true);
 txt(s,'Flipping the color comparison changes p(FAKE) by '+n(c.evidence.find(e=>e.type==='cross_modal'&&e.title.includes('Color')).impact)+'.',67,611,1125,52,25,C.muted);
}
// 10 Explanation evaluation.
{
 const f=data.faithfulness;
 const s=slide('Explanation sensitivity on 48 test records',f.protocol+' '+f.image_region_method+' '+f.cross_modal_method);
 chart(s,['Attribute token','Filler token'],[{name:'Mean absolute probability effect',values:[f.mean_absolute_attribute_token_effect,f.mean_absolute_filler_token_effect],fill:C.teal,valuesFormatCode:'0.000'}],59,170,629,354);
 txt(s,'0.924',764,186,400,86,62,C.teal,true);
 txt(s,'mean FAKE score drop\nafter caption correction',766,284,440,85,27);
 txt(s,'0.927',764,407,400,86,62,C.teal,true);
 txt(s,'mean FAKE score rise\nafter a color conflict',766,505,440,88,27);
 txt(s,'Both direction checks pass for every evaluated applicable case. This measures local sensitivity, not human explanation quality.',65,615,1144,60,23,C.muted);
}
// 11 Difficult analysis.
{
 const s=slide('Difficult examples and observed limits',data.difficult_case_selection+' The complete 16-case analysis is in results/REPORT.md and the paper. These are relatively low-margin cases rather than classification errors. All remain confident because the controlled task is simple. Sanity checks include missing image, blank/noise image, missing caption, negation and unsupported news text.');
 const chosen=[data.difficult_cases[0],data.difficult_cases[5],data.difficult_cases[7],data.difficult_cases[9]];
 table(s,['Case','Ground truth','Prediction','p(FAKE)'],chosen.map(c=>[c.id,c.label,c.prediction,n(c.fake_probability)]),65,169,1146,255,[501,227,227,191]);
 txt(s,'16 smallest-margin cases include 8 examples from each class. All are correct, so this run contains no test errors to explain.',67,468,1126,90,28);
 txt(s,'Unsupported language, natural backgrounds and mixed objects remain outside the demonstrated scope.',67,602,1126,66,25,C.muted);
}
// 12 Conclusion and next research.
{
 const s=slide('Conclusion and next experiments',refs+'\nThese references motivate future work only. They are not training sources for this prototype. Reproduce with python scripts/generate_dataset.py then python scripts/run_experiments.py. Start the local app with python app.py.');
 txt(s,'Explicit caption-image interactions improve performance on this controlled relational task.',66,167,1134,119,37,C.teal,true);
 txt(s,'A stronger study needs licensed natural data, pretrained baselines, source-separated evaluation, repeated seeds and calibration.',67,331,1129,110,29);
 txt(s,'Reproducibility',67,484,1100,48,27,C.ink,true);
 txt(s,'The repository contains the generator, trained parameters, all measured results and the local demonstration.',67,543,1126,81,27);
 txt(s,'Background: Fakeddit [1], FaceForensics++ [2], CLIP [3]',67,650,1100,31,20,C.muted);
}

const candidatePath=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
let finalPath=path.join(out,'VeriFuse_Presentation.pptx');
try{await fs.access(finalPath);finalPath=path.join(out,'VeriFuse_Presentation_'+Date.now()+'.pptx');}catch{}
const result=await finalizePresentation({workspaceDir:root,candidatePath,finalPath,pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[2,3,4,6,7,8,11].flatMap(i=>['--require-native-table-slide',String(i)])],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[2,3,4,6,7,8,11],requiredNativeChartOwnerSlides:[6,10],fontPolicy:{basis:'design',families:[family]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(qa,'validation.json')});
for(let index=0;index<p.slides.items.length;index++){
 const s=p.slides.items[index];
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(index+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(path.join(build,`slide-${index+1}.layout.json`),await layout.text());
}
console.log(JSON.stringify({finalPath,slides:p.slides.items.length,font:family,validation:result},null,2));
