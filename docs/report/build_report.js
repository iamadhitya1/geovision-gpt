const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  LevelFormat, convertInchesToTwip, PageBreak, Footer, PageNumber,
  ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, VerticalAlign, ShadingType,
} = require("docx");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size").default;

const body = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, ...opts })],
  spacing: { after: 200 },
});

const h1 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
const h2 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });

const bullet = (text) => new Paragraph({
  text,
  numbering: { reference: "main-bullets", level: 0 },
  spacing: { after: 140 },
});

const numbered = (text) => new Paragraph({
  text,
  numbering: { reference: "main-numbers", level: 0 },
  spacing: { after: 140 },
});

const ref = (text) => new Paragraph({ text, spacing: { after: 120 }, style: "reference" });

// ---- Figures ----
const FIG_DIR = path.join(__dirname, "figures");
let figureCount = 0;

function figure(filename, displayWidth, captionText) {
  figureCount += 1;
  const filePath = path.join(FIG_DIR, filename);
  const data = fs.readFileSync(filePath);
  const dims = sizeOf(data);
  const displayHeight = Math.round((displayWidth * dims.height) / dims.width);
  const ext = path.extname(filename).slice(1);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [
        new ImageRun({
          type: ext === "jpg" ? "jpeg" : "png",
          data,
          transformation: { width: displayWidth, height: displayHeight },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [
        new TextRun({ text: `Figure ${figureCount}. ${captionText}`, italics: true, size: 19 }),
      ],
    }),
  ];
}

// ---- Tables ----
const CELL_BORDER = { style: BorderStyle.SINGLE, size: 2, color: "B0B0B0" };
const CELL_BORDERS = { top: CELL_BORDER, bottom: CELL_BORDER, left: CELL_BORDER, right: CELL_BORDER };

function cell(text, opts = {}) {
  return new TableCell({
    borders: CELL_BORDERS,
    verticalAlign: VerticalAlign.CENTER,
    width: opts.width ? { size: opts.width, type: WidthType.PERCENTAGE } : undefined,
    shading: opts.header
      ? { type: ShadingType.CLEAR, fill: "1F2933" }
      : opts.shade
        ? { type: ShadingType.CLEAR, fill: "F2F4F5" }
        : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: opts.center === false ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [
          new TextRun({
            text,
            bold: !!opts.header || !!opts.bold,
            color: opts.header ? "FFFFFF" : undefined,
            size: 20,
          }),
        ],
      }),
    ],
  });
}

function dataTable(headerRow, rows, colWidths) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headerRow.map((t, i) => cell(t, { header: true, width: colWidths[i] })),
      }),
      ...rows.map(
        (r, ri) =>
          new TableRow({
            children: r.map((t, i) => cell(t, { width: colWidths[i], shade: ri % 2 === 1, center: i > 0 })),
          })
      ),
    ],
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 100, after: 220 },
    children: [new TextRun({ text, italics: true, size: 19 })],
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "main-bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.4), hanging: convertInchesToTwip(0.2) } } } }],
      },
      {
        reference: "main-numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.4), hanging: convertInchesToTwip(0.2) } } } }],
      },
    ],
  },
  styles: {
    paragraphStyles: [
      { id: "reference", name: "Reference", basedOn: "Normal", next: "Normal",
        run: { size: 21 }, paragraph: { spacing: { after: 120 }, indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.3) } } },
    ],
  },
  sections: [{
    properties: {},
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT] })],
        })],
      }),
    },
    children: [
      // ---------------- Title page ----------------
      new Paragraph({ text: "", spacing: { after: 400 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
        children: [new TextRun({ text: "Institute of Infrastructure Technology Research and Management (IITRAM)", bold: true, size: 26 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: "Department of Computer Engineering", size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
        children: [new TextRun({ text: "B.Tech Project Report", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 },
        children: [new TextRun({ text: "Multi-Modal Visual Question Answering (VQA) Modeling on Satellite and Earth Observation Imagery", bold: true, size: 28 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
        children: [new TextRun({ text: "Submitted by", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "M. Adhitya", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: "231049012001", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
        children: [new TextRun({ text: "Faculty Guide", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: "Dr. Ashish Soni", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: "Academic Year 2026-27", size: 22 })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- Abstract ----------------
      h1("Abstract"),
      body("Satellites and drones capture huge amounts of imagery, but turning a picture into a useful answer — how much of this area is flooded, is there a road near this river, how many buildings are in this scene — still usually needs a specialist and a purpose-built tool for each question. Visual Question Answering, or VQA, is a way around that: a system is shown an image and asked a plain-English question about it, and it answers in plain English back. VQA works well for everyday photos, but satellite images are different enough — taken from directly overhead, covering huge areas, full of small repeated objects, described with words an everyday photo captioner never needs — that it does not transfer over automatically."),
      body("This report explains the research this project is built on, and what has actually been done so far. Instead of taking a very large, already-built AI model and adapting it, this project builds a small VQA model from the ground up — choosing its parts, training it on real data, and testing what it can do. The model follows a design called LiT-4-RSVQA, published by Hackel et al. [6], which pairs a small image-understanding model with a small text-understanding model. This direction was chosen after the faculty guide reviewed the project's first plan and asked for something different: real, demonstrated understanding of how a VQA model works on the inside, built and trained by hand, rather than a wrapper around an existing large model. On top of reproducing the published design, this project also tries one genuine change of its own — swapping out a simple way of combining the image and the question for a more elaborate one — to see whether that actually improves the results. Both versions were first trained and tested on the institute’s HPC cluster on a single dataset: the reproduced baseline reached 75.16% average accuracy on held-out data, and the more elaborate version came out effectively level with it (72.33% overall accuracy against the baseline’s 72.74%) — an honest result, described in Section 4.4, along with a first attempt that failed because of a design flaw found and fixed along the way (Section 4.3)."),
      body("A further evaluation then asked a harder question: does either model work beyond the single dataset it was trained on, or only on images that closely resemble it? A robustness study found a real weakness — simply reducing an uploaded image’s brightness dropped the baseline’s rural/urban accuracy from 92% to 62%, close to the level of guessing without looking at the image at all. Both models were therefore retrained on RSVQA-LR together with a second, independently-collected dataset (RSVQA-HR, aerial imagery at a different resolution, with its own unseen-city test set), combined with image augmentation. The retrained models lose almost nothing on the original benchmark, answer most questions correctly about a city neither model saw in any form during training, and the brightness-related weakness is now largely fixed — the full method and results are in Section 4.5."),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- 1. Introduction ----------------
      h1("1. Introduction"),
      body("Satellite and aerial imagery is used for land mapping, disaster response, city planning, farming, and tracking environmental change. Right now, getting information out of that imagery usually means building a separate tool for each job: one system to detect objects, another to classify land type, another to spot what changed between two dates. Each of these tools needs an expert to set up and interpret, which limits how many people can actually make use of the data."),
      body("Visual Question Answering offers a simpler front door. A VQA system is given a picture and a question in ordinary language — “How many buildings are in this image?” or “Is there a road next to the river?” — and it answers in ordinary language too. Instead of building a new tool for every possible question, one VQA system can, in principle, answer many different questions about the same image. That makes the information usable by people who are not computer vision experts — a disaster response coordinator, a city planner, a farmer — who need an answer, not a technical pipeline."),
      body("Satellite VQA is not simply “VQA, but on a different kind of photo.” A satellite image is taken from directly overhead rather than from eye level, so a single frame can span kilometres and contain hundreds of small, similar-looking objects — buildings, roads, vehicles — instead of one clear subject. The vocabulary is different too: questions tend to be about counting and measuring rather than naming. Because of this gap, most recent research has responded by building large, dedicated AI systems for remote sensing: a pretrained image-understanding component, a small connecting piece, and a very large language model, all fine-tuned together on satellite-specific data. These systems work well, but they are expensive to build and run — often needing several billion adjustable internal values (called parameters) and multiple powerful GPUs just to train. That scale does not fit a single student working within one semester on a shared university GPU, and more importantly, working with a system that large does not require or prove that the student actually understands how the underlying model works — it can be adapted and used without ever being fully understood."),
      body("This report covers the background research behind this project, and the plan that came out of it, including one significant change of direction made partway through after feedback from the faculty guide. Section 2 covers the relevant published research, both general VQA and remote-sensing-specific work, ending with the specific small model this project follows and a look at what else has happened in the field recently. Sections 3 through 6 lay out the problem being addressed, how the project is built, what it is expected to produce, and exactly what has been done so far versus what is still ahead."),

      // ---------------- 2. Literature Review ----------------
      h1("2. Literature Review"),

      h2("2.1 Foundations of Visual Question Answering"),
      body("Kim et al. [1] survey the general VQA field and sort existing approaches into three groups. The earliest approach, called Joint Embedding, processes the image with one model (a CNN, a kind of network good at recognising visual patterns) and the question with another (an RNN, a kind of network that reads text word by word), then simply mixes the two results together to guess an answer — this tends to pull in irrelevant parts of the image as noise, since nothing tells the model which parts actually matter. The second group, Attention Mechanism methods, fix this by first learning which parts of the image and which words in the question are actually relevant, and only then combining them; Anderson et al.’s “Up-Down” model, which pairs object detection with this kind of attention, is named as an early influential example. The third group, Model-Agnostic methods, do not touch the model’s design at all — instead they change the training data or the training process to stop the model from cheating. VQA models are known to sometimes answer purely from patterns in the question and ignore the image entirely — for example, always answering “yellow” to “what colour is the banana” regardless of what is actually in the picture — and this group of methods tries to correct that habit."),
      body("The survey also covers the datasets used to train and test VQA systems, split into three kinds: everyday-photo datasets (VQA v2, VizWiz, OK-VQA), synthetic datasets built to test pure logical reasoning rather than visual recognition (CLEVR), and datasets deliberately built to stop a model from succeeding just by guessing the most common answer (VQA-CP, GQA). The standard scoring method, VQA Accuracy, counts an answer correct if it matches at least three out of roughly ten answers given by different human annotators for that question. The survey also notes that the field has recently moved toward very large, general-purpose models (such as BEiT-3 and Florence-VL) that can answer visual questions as just one of many things they do, often without being trained specifically for VQA at all — this project deliberately goes a different way, for reasons explained in Section 3."),

      h2("2.2 Vision-Language Modeling in Remote Sensing"),
      body("Weng et al. [2] survey AI models that combine images and text specifically for satellite and aerial imagery, and group them into three kinds by how they connect the image and the text during training. Contrastive models (such as GeoRSCLIP, RemoteCLIP, and SkyCLIP), following the well-known CLIP recipe, train an image-understanding model and a text-understanding model side by side so that a satellite image ends up mathematically “close to” its correct caption — this works well for quickly sorting or searching images, but is not built to hold a conversation. Instruction-tuned models take a pretrained image-understanding model, add a small connecting piece, and attach a large language model, then fine-tune the whole thing on (image, question, answer) examples so it learns to answer questions the way an assistant would — this is the recipe behind GeoChat and the other systems below, and it is the recipe this project deliberately does not follow, choosing a much smaller, purpose-built design instead. The third kind, image-generation models (such as DiffusionSat), create new satellite images from a text description and are not directly relevant to answering questions about existing ones."),
      body("The survey names RSGPT, GeoChat, LHRS-Bot, EarthGPT, and SkyEyeGPT as the leading instruction-tuned remote-sensing systems, and identifies RSVQA-HR and RSVQA-LR as the standard test sets used to measure and compare them — RSVQA-LR is the exact dataset this project trains and tests its own model on (Section 2.3). The survey closes by naming five open problems in the field, including handling vague or multi-step questions, getting models to explain their reasoning rather than just stating an answer, and the fact that most existing test sets are small or cover only a narrow range of question types — this last point is directly relevant to this project’s own choice of a small, well-understood dataset over a larger but less thoroughly documented one (see Section 3.1)."),

      h2("2.3 RSVQA: The Foundational Dataset and This Project's Training Data"),
      body("Lobry et al. [3] were the first to apply VQA to satellite imagery, and their paper remains the most-cited starting point for this area of research. Their main contribution was a way to build a VQA dataset automatically, using existing map annotations from OpenStreetMap instead of paying humans to write questions and answers by hand. The questions fall into five kinds: Count (“how many roads are present”), Presence (“is there a small retail place”), Comparison (“are there more buildings than roads”), Area, and a Rural/Urban classification. Using this method, they built two datasets: a Low Resolution set from Sentinel-2 satellite images over the Netherlands (772 images, 77,232 question-answer pairs, each pixel covering about 10 metres on the ground), and a High Resolution set from aerial photos over the north-eastern United States (10,659 images, 1,066,316 question-answer pairs, much sharper at about 15cm per pixel). The two trade off differently: the low-resolution set covers a wide area and can be refreshed often, but cannot make out small objects; the high-resolution set can, but is more expensive to capture and update."),
      body("This project trains and tests its model on the low-resolution set, RSVQA-LR, specifically. Its released files have an easy-to-miss quirk: each split’s file (train, validation, or test) actually contains every question from all three splits, with a flag marking which ones truly belong to that split — the other entries are inactive placeholders. This was checked directly against the real file contents rather than assumed, and doing it correctly recovers exactly 57,223 training, 10,005 validation, and 10,004 test question-answer pairs, adding up to the documented total of 77,232. The original paper’s own model was simple by today’s standards: a CNN (ResNet-152) reads the image, an older kind of text model (skip-thought vectors, built on an RNN) reads the question, the two results are multiplied together, and a small final layer predicts the answer. Even though far more accurate models have since been built, RSVQA-LR itself is still the standard benchmark nearly every later remote-sensing VQA system — including this project’s own — is measured against."),

      h2("2.4 GeoChat: A Unified Conversational Model for Remote Sensing"),
      body("Kuckreja et al. [4] present GeoChat, described as the first general-purpose, conversational AI model for remote sensing. It is built by taking an existing large model called LLaVA-1.5 (itself built on the Vicuna language model) and fine-tuning it — using a lighter-weight technique called LoRA that adjusts only a small fraction of the model’s internal values instead of all of them — on a purpose-built set of roughly 318,000 examples. What sets GeoChat apart from earlier remote-sensing systems is that it can answer questions about one specific part of an image (given as a bounding box) and point to exact object locations in its answers, rather than only describing a scene as a whole."),
      body("GeoChat is included here to show what a large-scale system looks like, not as something this project is trying to match or improve on. It is a multi-billion-parameter model that answers in free-form sentences; this project’s own model, by contrast, has about 8.2 million parameters and simply picks the best answer from a fixed list, trained fully rather than with the lighter LoRA technique. The two sit at very different points on the trade-off between accuracy, cost, and how deeply one person can actually understand what is happening inside — discussed further in Section 3."),

      h2("2.5 RSGPT: Data Quality Over Data Quantity"),
      body("Hu et al. [5] argue that the biggest obstacle to building strong remote-sensing AI models is not the model’s design but a lack of high-quality, carefully written training data — most existing datasets are either generated automatically or use very short, thin captions. Motivated by the finding that fine-tuning a language model on a small but carefully made dataset can match results from a much larger, messier one, they built RSICap: 2,585 captions written entirely by human annotators, each describing a scene in real detail (colour, shape, quantity, position). They also built a companion test set, RSIEval, combining captions and question-answer pairs, which has since become a standard way to evaluate later systems, including GeoChat."),

      h2("2.6 LiT-4-RSVQA: The Architecture This Project Follows"),
      body("Hackel et al. [6] ask a question directly relevant to this project: can a VQA model with only a few million parameters — a small fraction of what GeoChat-style systems use — still perform well? Their design pairs BERT-Tiny, a much-shrunk version of the well-known BERT text model (2 layers and 128 internal dimensions, instead of BERT’s usual much larger size), with a small image-understanding model — they tried three options, XCiT-Nano, DeiT-Tiny, and MobileViT-S, all under 12 million parameters. The image and text results are combined with a simple elementwise product (multiplying the two result-lists position by position) and passed to a small classifier that picks the single most likely answer out of the 1,000 most common answers seen during training. Their best combination — a pretrained XCiT-Nano paired with BERT-Tiny — reached 64.61% average accuracy while using only 8.1 million parameters, actually beating a much bigger 277-million-parameter model (VBFusion) by about 9 percentage points."),
      body("The single most useful finding in the paper for this project is a direct comparison of two training strategies using the identical architecture: starting the image model from pretrained weights (meaning it already knows something about recognising general visual patterns before training on satellite images begins) versus training it completely from scratch. Starting from pretrained weights reached 64.61%; starting from scratch reached only 51.71% — nearly 13 percentage points lower, on the exact same architecture. That gap is the direct evidence behind this project’s own decision to start from pretrained models and fine-tune them, rather than training everything from a blank slate: the published evidence shows this is not a shortcut, it is simply the better-performing choice. This project reproduces the paper’s best configuration — pretrained XCiT-Nano with the simple elementwise combination — on RSVQA-LR as a working baseline, and then builds a second version that replaces the simple combination step with a more elaborate one, described in Section 4.2, as this project’s own original contribution."),

      h2("2.7 Recent Developments (2024–2026)"),
      body("A literature search carried out in September 2026, after this project’s direction changed, turned up work published since the LiT-4-RSVQA paper — useful context for understanding where the field stands today, though it does not change this project’s own plan. The largest systems in the field have kept getting bigger rather than smaller: Lin et al. [7] introduce RS-MoE, which splits work across several smaller specialist language models rather than using one big one; Liu et al. [8] introduce Co-LLaVA, which combines two separate AI models together and outperforms GeoChat and SkySenseGPT on the RSVQA-LR dataset this project also uses; and Liu et al. [9] introduce HSF-Net, which adds an extra processing step on top of LLaVA-1.5 to help it notice fine spatial detail it would otherwise miss. All three are still billion-parameter systems, mentioned here to show what current large-scale research looks like, not as something this project attempts to match in size."),
      body("One result from this recent work directly supports the idea behind this project: Lin et al. [7] report that their 1-billion-parameter RS-MoE model performs about as well as models with 13 billion parameters — independent, up-to-date proof that a smaller, more carefully built model can hold its own against a much larger one, echoing what LiT-4-RSVQA already showed at an even smaller scale. On the question of test datasets, Wang et al. [11] describe RSVQA plainly as covering “basic statistics for general objects,” and Zi et al. [10] have since released a much larger benchmark, RSVLM-QA, with 162,373 question-answer pairs compared to RSVQA-LR’s 77,232. This project’s choice to use the smaller RSVQA-LR dataset is therefore a deliberate scope decision — it is small enough to fully train and test within the time and computing power actually available — rather than something overlooked; moving to a larger dataset is named as a direction for future work in Section 5. Zhang et al. [12] present closely related prior work that also uses the RSVQA family of datasets, though with a different combination technique (a multi-step attention mechanism) than this project’s comparison. Finally, Lu et al. [13] review how different AI models combine image and text information using attention, giving this project’s own combination-method comparison (Section 4.2) a grounding in existing research rather than being an arbitrary choice."),

      h2("2.8 Summary"),
      body("Taken together, the research reviewed above shows two separate trends running in parallel: a shift from RSVQA’s simple, small model (2020) toward large, conversational, instruction-following systems (2023–2026) that can hold a dialogue and point to exact locations in an image, and a smaller, separate line of work — leading up to LiT-4-RSVQA — asking whether a much smaller, fully-trained model can still hold its own. This project follows the second path, for the reasons set out in Section 3: it fits within the computing resources actually available to one student on a shared university GPU cluster, and building it from the ground up, rather than adapting an existing large system, is a real and demonstrable way to show an actual working understanding of how a VQA model functions — not just how to call one."),

      // ---------------- 3. Problem Statement and Objectives ----------------
      h1("3. Problem Statement and Objectives"),
      body("Note: this section reflects a change of direction made after the faculty guide reviewed the project's first plan. That original plan involved adapting a large existing AI model and adding a separate verification layer on top of it. That plan is no longer being pursued, for the reasons explained below.", { italics: true }),

      h2("3.1 Problem Statement"),
      body("The large, instruction-following systems covered in Section 2 (see Sections 2.2 and 2.4) get strong results, but at a scale — typically several billion adjustable values, trained on multiple powerful GPUs at once — that is simply too large for one student’s share of a university GPU cluster within a single semester. More importantly, working with a model that large does not require, or prove, an actual understanding of how it works: it can be adapted through an existing interface or lightly fine-tuned without the person doing it ever needing to understand its internal mechanics. This project instead addresses a more achievable question: can a small, fully-understood, fully-trained VQA model — following the published evidence in Hackel et al. [6] that small, pretrained models can remain competitive — be built, trained, and tested end to end by one student, and does a genuine design change on top of it (replacing the simple combination step with a more elaborate one) actually make it more accurate?"),

      h2("3.2 Objectives"),
      numbered("Study existing VQA research, both for everyday photos and for satellite imagery specifically, including research on small, efficient models (completed; see Section 2)."),
      numbered("Choose and justify a small VQA design that can realistically be trained and tested using the computing power available on the institute’s shared HPC GPU cluster, based on current published research (completed; see Sections 2.6–2.7)."),
      numbered("Build the data pipeline (loading RSVQA-LR correctly, including its non-obvious file format), the model itself (a small text model plus a small image model, a simple combination step, and a final answer-picking layer), and the training and testing code — all written so it works the same whether run on a CPU or a GPU, and fully checked on a CPU before ever touching the GPU cluster (completed)."),
      numbered("Train and test the model on the institute’s HPC GPU cluster, using its job-scheduling system, to reproduce LiT-4-RSVQA’s own best result on RSVQA-LR as a working baseline (completed 8 September 2026: 72.74% overall accuracy, 75.16% average accuracy on the true held-out test split, closely matching the published architecture’s parameter count — see Section 4.4 for the full result and Section 4.3 for the real technical problems found and solved to get there)."),
      numbered("Build and test a version of the model that replaces the simple image-question combination step with a more elaborate one, using the exact same data and starting point, as this project’s own original contribution (completed 21 September 2026: after a first attempt failed because of a design flaw that was found, proven, and fixed — see Section 4.3 — the corrected version reached 72.33% overall accuracy and 72.29% average accuracy, effectively level with the baseline; see Section 4.4)."),
      numbered("Test both models beyond the single dataset they were trained on, and, if a genuine weakness is found, fix it rather than only report it (completed 22–24 September 2026: a robustness study found both models were brittle to simple image changes such as reduced brightness; retraining both on a second, independent dataset with image augmentation fixed the main weakness and produced measurable generalisation to a city neither model ever saw in training — see Section 4.5)."),

      // ---------------- 4. Proposed Methodology ----------------
      h1("4. Proposed Methodology"),
      body("Note: this section reflects the current, revised project direction, replacing the original plan first presented to the faculty guide.", { italics: true }),

      h2("4.1 System Architecture"),
      body("The model follows Hackel et al. [6] directly. It has two starting components, each already pretrained on other data before this project’s own training begins: a text-understanding component (BERT-Tiny — specifically the official Google-released checkpoint google/bert_uncased_L-2_H-128_A-2, checked to match the paper’s exact specification of 2 layers, 128 internal dimensions, and 2 attention heads), and an image-understanding component (a small pretrained image model, with XCiT-Nano used as the main choice since it performed best in the original paper, and two alternatives, DeiT-Tiny and MobileViT-S, also implemented). Both components are fine-tuned during training rather than kept frozen or trained from a blank slate, following the evidence discussed in Section 2.6. Their two outputs are then combined — either by a simple elementwise product in the reproduced baseline, or by a more elaborate combination step (described below) in this project’s own experimental version — and the combined result is passed to a small final layer that picks the most likely answer out of the 1,000 most common answers seen in training — a multiple-choice design rather than one that writes free-form sentences. This is a deliberate choice, not a shortcut: a model able to write any sentence needs several billion adjustable parameters and correspondingly more training data and compute, which this project’s available GPU time on a shared university cluster does not support; picking from a fixed list is what lets a model of this size be trained fully, by one person, within a semester. The baseline version of the whole model has 8,217,896 adjustable parameters in total, closely matching the 8.1 million reported in the original paper; the experimental version, which adds the more elaborate combination step, has 9,269,544."),
      ...figure(
        "fig-architecture.png",
        560,
        "System architecture. The image and text encoders each produce their own pretrained representation; the fusion step (elementwise product or cross-attention) is the one component swapped between the reproduced baseline and this project's own experimental version; a small classifier then scores 1,000 possible answers."
      ),
      body("The training and testing code is written so the exact same code works on a plain laptop CPU or on a proper GPU without any changes — a single function checks whether a GPU is available and uses it automatically, falling back to the CPU otherwise. This let the whole pipeline — loading the data, preparing the text and images, running one training step, saving a checkpoint file, and testing the model’s accuracy — be fully checked and debugged on the development laptop’s CPU before any of it touched the institute’s shared GPU cluster, so that GPU time was not spent finding basic bugs."),
      body("A separate piece of software was also built earlier in the project: a working website (covering user accounts and permissions, plus uploading and managing satellite images). It is built with a clear separation between its different internal layers, checked automatically so that a lower-level part of the code can never accidentally depend on a higher-level one. Once the trained model produced a working checkpoint, this website was extended with a third layer that actually serves the model: given an uploaded image and a plain-English question, it returns an answer, its confidence, and — since the whole point of this project’s own experiment is comparing two versions of the model — both trained versions’ answers side by side when both are available, not just one. This is real, working, tested software, and it is how the trained model is actually meant to be used and demonstrated, not only reported on in numbers. A visual front end, an interactive map, and a results dashboard were then built on top of it (Section 6.2), so the whole system can be shown live."),

      ...figure("01-signin-fixed.jpg", 480, "The application's sign-in page."),
      ...figure(
        "04-workspace-answer.jpg",
        480,
        "The question workspace: both trained models answer the same question about the same held-out image side by side, each with a confidence score and runner-up answers."
      ),

      h2("4.2 Original Contribution: Comparing Two Ways to Combine Image and Question"),
      body("This project’s own original contribution is a direct, fair comparison between two different ways of combining the image and the question, using identical data, identical starting components, and identical training settings — changing only the one thing being tested. The baseline reproduces LiT-4-RSVQA’s own method: both the image result and the question result are resized to the same length and then multiplied together position by position, a simple but effective combination. The experimental version replaces this with a cross-attention step, in which every word of the question is allowed to directly “look at” every small patch of the image and decide which patches matter most, before the two are combined — a more elaborate mechanism reviewed generally in Lu et al. [13]. Because everything else about the two versions is kept identical, any difference in the final accuracy can be attributed specifically to this one change. This is meant to be a genuine, honestly reported comparison: if the more elaborate combination step does not actually do better than the simple one, that is still a real and useful result, not a failed experiment."),

      h2("4.3 Development Workflow"),
      body("The website (Section 4.1) was built one piece at a time, with each part explained, designed, built, tested, and connected to the rest before starting the next, following the faculty guide’s original instruction. The first two parts — user accounts/permissions and satellite image handling — were built first and are complete, backed by over a hundred automated tests, with the layering rule mentioned above checked automatically on every change. The model-serving layer, the visual front end, the map view, and the dashboard followed once a trained model existed (Section 6.2). The remaining parts of the original ten-part plan assumed a different, larger kind of model (one that holds a conversation and generates free text) and were redesigned once the model actually being built turned out to be a much smaller classifier instead — described further in Section 6."),
      body("The model (Sections 4.1–4.2) was built following a “prove it works without a GPU first” approach: the data-loading code, the model itself, and the training and testing scripts were each written and checked on the development laptop’s own CPU, including a full run-through of the whole pipeline, before any of it ran on the institute’s GPU cluster. Getting access to that cluster required a signed form submitted to the coordinator of the institute’s AI/ML research centre, after which login details were issued by email. Jobs on the cluster can only be submitted through its job-scheduling system, called SLURM, from a separate “master” computer — running anything directly on the GPU computer itself is against policy and gets shut down by the system administrator if it happens. The scripts that submit jobs to SLURM were written and checked over before the very first real session on the cluster, so that the actual GPU time would be spent running an already-working job, rather than debugging basic mistakes on a shared, time-limited machine."),
      body("The first real session on the cluster (8 September 2026) turned up several real, cluster-specific problems that could not have been found by working on a CPU alone — each one was diagnosed by looking directly at what the cluster actually reported, not guessed at. First, the cluster’s four physical GPUs are set up so that three of them share one identical label as far as the job scheduler is concerned, meaning a plain request for “one GPU” could accidentally land on the one GPU the institute has reserved and asked students not to use. The fourth physical GPU is split into six smaller, separately-usable slices using a technology called MIG (Multi-Instance GPU), matching exactly what the coordinator’s email had described. Asking specifically for one of the small slices, by its exact name, turned out to be the only way to guarantee never landing on the reserved GPU by accident, confirmed directly against the cluster’s own reported hardware list before any job was run. Second, the cluster’s installed graphics driver only supports a certain version of the underlying GPU software toolkit (CUDA 12.4); a plain, default installation of the training software (PyTorch) pulled in a version built for a newer toolkit than the driver supports, and it silently ran on the CPU instead of the GPU with no clear error, only a warning easy to miss in a long log file — found by comparing that warning against the driver’s own reported version, and fixed by installing a matching version explicitly. Third, and closely related, the way the job scheduler identified which small GPU slice had been assigned did not, on its own, match the exact identifier the GPU software actually needed to use that slice — this again caused the job to silently fall back to running on the CPU, with no error message at all. This was fixed by having the job script look up the correct identifier for itself, from the GPU driver’s own device list, every time it runs. Fourth, the batch size used in the original published paper was tuned for a full, much larger GPU than the roughly 9.5 gigabytes available on the cluster’s small slice, and the very first attempt at real training ran out of memory; it was reduced to a smaller, working value."),
      body("With those four fixes in place, the baseline configuration trained cleanly for all ten epochs and was evaluated on the held-out test split, producing the result reported in Section 4.4. The comparison experiment (Section 4.2) was then run the same way. Its first attempt failed, for a reason worth explaining, described below. Separately, the website’s new model-serving layer (Section 4.1) was built and tested on the development laptop; two further, unrelated packaging problems were found and fixed while wiring the trained model’s code into the website’s own codebase so the two could share it without duplicating it — a directory naming clash with a reserved convention in Python’s packaging tooling, and an installation step that was unintentionally bundling the entire training dataset into the website’s own installable package. Both were caught by inspecting the built package’s actual contents rather than assuming the build had done the right thing, and both are now fixed. A third, related problem surfaced later: the website had installed a frozen copy of the model’s code rather than a live link to it, so the corrected model design initially never reached the website. This was fixed by making the link live, so the two can no longer drift apart."),

      body("The first version of the comparison experiment scored only 36.06% overall accuracy on the test split (27.07% average) — about half the baseline — and its training never improved: validation accuracy stayed at the same one or two values from the first epoch to the last. That pattern suggests a model that gives nearly the same answer whatever it is asked, so the cause was investigated rather than the number being reported as a finding about cross-attention. The cause was a flaw in this project’s own design of the step. The image and the question had each been squashed down to a single list of numbers before being combined, so the “look at the image” step had only one thing to look at — and when there is only one thing to look at, the weighting is always exactly 100% no matter what the question says. The question therefore had no influence at all on the model’s output. This was confirmed directly: changing the question changed the step’s output by exactly zero, while changing the image changed it. The fix was to keep the image and the question as sequences of pieces and let every word of the question look at all of the image’s patches. The model’s structure was reworked so that each combination method is chosen in one place, instead of being swapped in separately by the training, testing, and website code (which is how the flaw had gone unnoticed), and an automated test now checks that both the question and the image genuinely change a combination step’s output. The corrected version was retrained on the cluster and tested (Section 4.4); the first attempt’s saved files were kept as evidence. The episode is a useful reminder that a training run that finishes without errors is not evidence that the model is doing what it was designed to do."),

      h2("4.4 Evaluation Plan"),
      dataTable(
        ["Test accuracy (RSVQA-LR, 9,838 Qs)", "Baseline (elementwise)", "Cross-attention", "No image (guess)"],
        [
          ["Overall", "72.74%", "72.33%", "70.42%"],
          ["Average (paper's metric)", "75.16%", "72.29%", "64.84%"],
          ["Rural / urban (100 Qs)", "92.0%", "81.0%", "56.0%"],
          ["Presence (2,955 Qs)", "90.7%", "90.2%", "88.3%"],
          ["Comparison (4,002 Qs)", "90.4%", "89.2%", "87.1%"],
          ["Counting (2,781 Qs)", "27.5%", "28.8%", "28.0%"],
        ],
        [40, 20, 20, 20]
      ),
      tableCaption("Table 1. Held-out test accuracy, original single-dataset models. \"No image\" is a predictor that never sees the picture and answers each question with its most common training answer for that exact wording."),
      body("The trained model is tested on RSVQA-LR’s held-out validation and test data — data it never saw during training, used specifically to check how well it generalises. Four numbers are recorded: overall accuracy (the plain percentage of questions answered correctly), average accuracy (the accuracy for each of the four question types — presence, count, comparison, and rural/urban — averaged together equally, matching exactly how the original LiT-4-RSVQA paper reports its own results, so this project’s numbers can be compared directly against a published result), accuracy broken down by question type individually, and the total parameter count."),
      body("The baseline (simple combination) configuration has been trained and tested. On the true held-out test split (9,838 examples, after 166 examples with answers outside the 1,000-answer vocabulary were excluded, consistent with the vocabulary being built from the training split only), it reached 72.74% overall accuracy and 75.16% average accuracy — above the paper’s own reported 64.61%, though this is not a strictly like-for-like comparison, since the published result was obtained on a much larger training set (RSVQAxBEN, roughly 15 million examples derived from BigEarthNet-S2) rather than RSVQA-LR directly. Accuracy by question type was uneven: 92.0% on rural/urban, 90.7% on presence, 90.4% on comparison, but only 27.5% on count — consistent with counting being the hardest question type for this kind of model across the wider VQA literature reviewed in Section 2, not a defect specific to this implementation. The experimental (more elaborate combination) version, trained and tested under identical conditions, reached 72.33% overall accuracy and 72.29% average accuracy on the same test split, with 9,269,544 parameters (about 13% more than the baseline). By question type it scored 81.0% on rural/urban, 90.2% on presence, 89.2% on comparison, and 28.8% on count, against the baseline’s 92.0%, 90.7%, 90.4%, and 27.5%. The two versions are effectively level overall, 0.4 percentage points (about 40 of the 9,838 questions) apart. The larger gap in average accuracy comes from the rural/urban type, which has only 100 test questions but counts for a quarter of the average: the baseline answering 11 more of those correctly moves the average by almost three points. On the three large question types the two versions are within about 1.3 points of each other either way, with the experimental version slightly ahead on counting. Each version was trained only once, so differences this small should not be read as evidence that either method is better. The honest conclusion is that, in this setup, the more elaborate cross-attention combination did not measurably improve on the simple baseline, while costing about 13% more parameters. Repeating each run with several different random starting points, to measure how much results vary from run to run, is named as future work."),

      body("A further check was made on how much of this accuracy comes from the question alone. RSVQA-LR’s questions follow fixed templates, so many answers can be guessed from the words without looking at the image at all. A simple predictor that never sees the image, and answers each question with the most common answer given to that exact question in the training data, scores 70.42% overall and 64.84% average accuracy on the same test split (56.0% on rural/urban, 88.3% on presence, 87.1% on comparison, and 28.0% on count). Both trained models therefore beat this no-image predictor by only about two points overall. The image clearly matters for rural/urban, where the models score 92.0% and 81.0% against 56.0% for the guess; it adds a little on presence and comparison; and it adds nothing on counting, where both models are level with always giving the most common count. This is a known weakness of templated benchmarks of this kind rather than a defect of this implementation, and it is the honest floor against which the models should be judged, not zero. It also sharpens the conclusion above: the cross-attention step could only show its benefit where the image matters, and even there it did not beat the simple baseline. The no-image predictor is included in the project’s dashboard and can be reproduced with one script."),

      h2("4.5 Robustness Study and Widening the Model for Generalisation"),
      body("A model that scores well on one held-out split of one dataset has only been shown to work on images that closely resemble its training data — it has not been shown to be robust, meaning that it keeps working when the input changes in ways that should not matter, or general, meaning that it works on data collected independently of its training set. Both properties matter more for a real deployment than a single test-split number, so both were measured directly rather than assumed."),
      body("The robustness study applied controlled changes to the same held-out test images and re-asked the same questions, comparing each altered answer against the answer on the unchanged image. Some changes cannot change the true answer (reducing brightness, halving the resolution, or mirroring or rotating an image for a question that does not refer to a direction, such as “is this rural or urban” rather than “is there a road on the left”); one change, zooming in, can change the true answer, so it was read differently. The result exposed a real weakness: reducing the baseline’s image brightness alone dropped its rural/urban accuracy from 92.0% to 62.0%, and halving the image’s resolution dropped it to 58.0% — close to the 56.0% a model gets by ignoring the image entirely. On the other, larger question types, accuracy barely moved only because most of it never came from the image in the first place, but a closer measure showed the same weakness: about 13% of the baseline’s answers changed under a mirror or a rotation that could not have changed the true answer, meaning the model was not looking at the scene the way a person would. Mirroring and rotating did not hurt rural/urban accuracy specifically, which showed the weakness was to the image’s appearance and scale, not to its geometry."),
      body("Both models were therefore retrained on RSVQA-LR together with a second dataset, RSVQA-HR (Lobry et al. [3], the same paper family as RSVQA-LR): about 1.07 million question-answer pairs over 15-centimetre aerial photography of the north-eastern United States, collected independently of RSVQA-LR’s 10-metre Sentinel-2 imagery of the Netherlands, and including its own held-out test set from a city, Philadelphia, that appears nowhere else in either dataset. Training also added label-preserving image augmentation — random changes to brightness, contrast, saturation, and resolution, and mirroring or rotating images for questions whose true answer does not depend on direction — so the model would see many appearances of the same underlying scene rather than one fixed one. A resumable training script was written so a job could save its exact progress and continue from the same point after being stopped by the cluster’s four-hour job time limit, verified by deliberately interrupting and resuming a run before it was trusted on the cluster."),
      body("On RSVQA-LR’s own test split, the retrained models cost almost nothing against the original single-dataset numbers in Section 4.4: the simple-combination version moved from 72.74% to 72.02% overall accuracy, and the cross-attention version moved from 72.33% to 71.96% — differences smaller than the gap between the two combination methods themselves. On RSVQA-HR’s own test cities, the two versions reached 81.9% and 80.4% overall accuracy; on Philadelphia, the city neither model saw during training in any form, they reached 77.2% and 74.4%. A gap of five to six percentage points between a dataset’s own test cities and a genuinely unseen city is a real but modest cost of generalising, not a collapse to guessing — and it is the actual evidence that these models are useful beyond a small, selected set of images from one dataset, for both combination methods."),
      dataTable(
        ["Test accuracy", "RSVQA-LR (9,838 Qs)", "RSVQA-HR, own cities (189,617 Qs)", "RSVQA-HR, Philadelphia — unseen city (90,112 Qs)"],
        [
          ["Elementwise — overall / average", "72.02% / 73.65%", "81.9% / 81.4%", "77.2% / 76.5%"],
          ["Cross-attention — overall / average", "71.96% / 73.91%", "80.4% / 79.9%", "74.4% / 73.8%"],
        ],
        [28, 24, 24, 24]
      ),
      tableCaption("Table 2. Accuracy of the widened models on three held-out test sets. Philadelphia never appears in training, in any form, for either dataset."),
      body("The robustness weakness itself was also directly re-measured, using the same perturbations on the same RSVQA-LR test images. For the simple-combination model, the same reduced-brightness test that had dropped rural/urban accuracy from 92.0% to 62.0% now costs nothing at all — 88.0% clean, 88.0% darkened — and halved resolution costs 3 points (85.0%) rather than 34. For the cross-attention model, the same darkening test improves from a 15-point drop (81.0% to 66.0%) to a 3-point drop (89.0% to 86.0%), though halved resolution still costs about 8 points, an honest partial result rather than a complete fix. Taken together, the widened training turned a measured weakness into a mostly-fixed one for both combination methods, and it confirms that this project’s central comparison — whether cross-attention measurably improves on the simple baseline — reaches the same conclusion (no measurable improvement) under a training setup materially different from the one it was first observed under, which is stronger evidence than either result alone."),
      ...figure(
        "fig-robustness-chart.png",
        480,
        "Rural/urban accuracy under image perturbation, before and after widening the training data (elementwise fusion). The darker-image drop, the clearest sign of the original weakness, shrinks from 30 points to essentially zero."
      ),

      // ---------------- 5. Expected Outcomes ----------------
      h1("5. Expected Outcomes"),
      body("Note: this section reflects the current, revised project direction.", { italics: true }),
      bullet("A fully built, trained, and tested small VQA model for satellite imagery, reproducing LiT-4-RSVQA’s published design and reaching 75.16% average accuracy on RSVQA-LR’s test split — achieved."),
      bullet("A second version of the same model, using a more elaborate way of combining the image and the question, trained and tested under identical conditions — giving a direct, honestly reported comparison against the published baseline. This is the project’s original contribution; it is complete, and the result was level with the baseline (72.33% against 72.74% overall accuracy)."),
      bullet("Evidence that both models work beyond the single dataset and test split they were first trained and measured on — achieved by finding a real robustness weakness (a dimmer image alone dropped rural/urban accuracy from 92% to 62%), retraining both models on a second, independently-collected dataset with image augmentation, and measuring the result: almost no cost on the original benchmark, correct answers on most questions about a city neither model ever saw in training, and the robustness weakness largely fixed (Section 4.5)."),
      bullet("A working, tested website that actually serves the trained model — upload an image, ask a question in plain English, and see the answer, with both trained versions shown side by side, plus a map view and a results dashboard — not just a report of results. Built and working, and updated to serve the retrained, widened models."),
      bullet("A project report describing the design, the training process, and the results, including an honest account of the mid-project change of direction, the robustness weakness found afterwards, and the reasons behind both."),
      bullet("A suggested direction for future work, identified during the literature review and confirmed by this project’s own results: extending the trained model, or its testing, to larger and more recent satellite-VQA datasets such as RSVLM-QA [10], and closing the remaining scale-sensitivity gap that the widened training did not fully fix (Section 4.5)."),

      // ---------------- 6. Module / Task Plan ----------------
      h1("6. Module and Task Plan"),
      body("The project now runs along two separate tracks: the model itself (the priority, following the faculty guide’s review) and the website that serves it. The faculty guide has since clarified that the website is not graded separately from the model, but is still expected as a working, demonstrable interface — a live demonstration is the norm for project evaluation at this institute, and a well-built interface is worth real effort, not treated as optional polish. Both tracks are listed here."),
      h2("6.1 Model Track"),
      bullet("Dataset — RSVQA-LR downloaded; its file-format quirk checked directly against the real files; the data-loading code written and tested. Complete."),
      bullet("Model architecture — BERT-Tiny plus a choice of small image models (XCiT-Nano, DeiT-Tiny, or MobileViT-S), the simple combination baseline and the more elaborate experimental version, and the final answer-picking layer; built and fully checked on a CPU (8,217,896 parameters for the baseline, closely matching the paper’s reported 8.1 million, and 9,269,544 for the cross-attention version). Complete."),
      bullet("Training and testing scripts — written to work the same on CPU or GPU, and fully checked end to end on a CPU. Complete."),
      bullet("Access to the institute’s HPC cluster — declaration form submitted, login details received, job-submission scripts written and checked. Complete."),
      bullet("The baseline training run — completed 8 September 2026 on the institute’s GPU cluster: 72.74% overall accuracy, 75.16% average accuracy on the test split. Complete (see Section 4.4 for the full result and Section 4.3 for the real technical problems found and fixed along the way)."),
      bullet("The comparison experiment (cross-attention combination) — complete. A first attempt failed because of a design flaw (found, proven, and fixed; see Section 4.3); the corrected version reached 72.33% overall and 72.29% average accuracy, level with the baseline (Section 4.4)."),
      bullet("Robustness study — complete. Found that both models were brittle to simple image changes such as reduced brightness (rural/urban accuracy dropped from 92% to 62% on the baseline); this motivated the widening work below (Section 4.5)."),
      bullet("Widening both models to a second dataset (RSVQA-HR) with image augmentation — complete, 22–24 September 2026. Both retrained models cost under one point on the original RSVQA-LR benchmark, generalise to a city neither saw in training (77.2% and 74.4% overall accuracy on an unseen city, against 81.9% and 80.4% on the second dataset’s own test cities), and the robustness weakness is largely fixed (dimmed-image rural/urban accuracy restored to 88% and 86%, from 62% and 66%). These are the models the website now serves (Section 4.5)."),
      bullet("Final results and write-up — the two combination methods compared against each other, against the published baseline, and against their own widened versions, in Sections 4.4 and 4.5. Complete; a final review remains."),

      h2("6.2 Website / Interface Track"),
      bullet("Module 1 — User accounts, permissions, and an activity log. Complete: over a hundred automated tests passing."),
      bullet("Module 2 — Satellite image handling (uploading, checking the file is genuinely the type it claims to be, reading its location data, generating preview thumbnails). Complete."),
      ...figure("03-library-uploaded.jpg", 460, "The image library after uploading three held-out test scenes."),
      bullet("Module 3/4 — Serving the trained model: given an uploaded image and a plain-English question, returns an answer with a confidence score, and both trained versions’ answers together when more than one is available, so the model comparison that is this project’s own contribution is visible directly in the interface, not only in this report. Complete: both trained versions are loaded and answer questions together, at roughly 0.4 seconds per question on a laptop CPU with one model loaded."),
      bullet("Module 5 — A visual front end: sign-in, an image library with drag-and-drop upload, and a workspace where a question is asked and both models’ answers appear side by side, with a warning when a model is close to guessing. Complete; checked at both desktop and phone widths."),
      bullet("Module 6 — An interactive map view: uploaded images that carry location data are drawn at their true position over a dark or satellite basemap, and can be clicked to open the question workspace. Complete."),
      bullet("Module 7 — A dashboard showing both models’ measured accuracy on the test split, broken down by question type with the number of test questions behind each figure, alongside live usage figures (questions asked, how often the two models agree, average confidence). Complete."),
      ...figure(
        "05-dashboard.jpg",
        400,
        "The live dashboard: measured accuracy by question type against the no-image floor, and real usage statistics from questions asked in this session."
      ),
      bullet("Live demonstration — a one-command launcher and a written runbook covering a walkthrough script, the questions that work best on held-out test images, and likely examiner questions with honest answers. Complete."),

      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- References ----------------
      h1("References"),
      ref("[1] B. S. Kim, J. Kim, D. Lee, and B. Jang, “Visual Question Answering: A Survey of Methods, Datasets, Evaluation, and Challenges,” ACM Computing Surveys, vol. 57, no. 10, Article 249, May 2025."),
      ref("[2] X. Weng, C. Pang, and G.-S. Xia, “Vision-Language Modeling Meets Remote Sensing: Models, Datasets, and Perspectives,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 276-323, Sept. 2025."),
      ref("[3] S. Lobry, D. Marcos, J. Murray, and D. Tuia, “RSVQA: Visual Question Answering for Remote Sensing Data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 12, pp. 8555-8566, Dec. 2020."),
      ref("[4] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “GeoChat: Grounded Large Vision-Language Model for Remote Sensing,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 27831-27840."),
      ref("[5] Y. Hu, J. Yuan, C. Wen, X. Lu, and X. Li, “RSGPT: A Remote Sensing Vision Language Model and Benchmark,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 224, pp. 272-286, Jun. 2025."),
      ref("[6] L. Hackel, K. Clasen, M. Ravanbakhsh, and B. Demir, “LiT-4-RSVQA: Lightweight Transformer-based Visual Question Answering in Remote Sensing,” arXiv:2306.00758, 2023."),
      ref("[7] H. Lin, D. Hong, S. Ge, C. Luo, K. Jiang, H. Jin, and C. Wen, “RS-MoE: A Vision–Language Model With Mixture of Experts for Remote Sensing Image Captioning and Visual Question Answering,” IEEE Transactions on Geoscience and Remote Sensing, 2024."),
      ref("[8] F. Liu et al., “Co-LLaVA: Efficient Remote Sensing Visual Question Answering via Model Collaboration,” Remote Sensing, 2025."),
      ref("[9] F. Liu et al., “Boosting Large Vision-Language Models With Hierarchical Semantic Fusion for Remote Sensing Visual Question Answering,” IEEE Geoscience and Remote Sensing Letters, 2026."),
      ref("[10] X. Zi et al., “RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering,” in Proc. 33rd ACM International Conference on Multimedia, 2025."),
      ref("[11] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “EarthVQANet: Multi-task visual question answering for remote sensing image understanding,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 212, pp. 422-439, 2024."),
      ref("[12] M. Zhang et al., “Multistep Question-Driven Visual Question Answering for Remote Sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, 2023."),
      ref("[13] S. Lu, M. Liu, L. Yin, Z. Yin, X. Liu, and W. Zheng, “The multi-modal fusion in visual question answering: a review of attention mechanisms,” PeerJ Computer Science, vol. 9, e1400, 2023."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("BTP_Report_VQA_RemoteSensing.docx", buf);
  console.log("written, bytes:", buf.length);
});
