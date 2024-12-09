import fs from "fs";
import path from "path";

import { AutoProcessor } from "../src/transformers.js";
import { load_cached_image } from "./asset_cache.js";
import { init, MAX_TEST_TIME } from "./init.js";
import { fileURLToPath } from "url";

// Initialise the testing environment
init();

// Collect all unit tests, which can be found in files of the form:
// `tests/models/<model_type>/test_image_processors_<model_type>.js`
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const models_dir = path.join(__dirname, "models");
const model_types = fs.readdirSync(models_dir);
for (const model_type of model_types) {
  const dir = path.join(models_dir, model_type);

  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    continue;
  }

  const file = path.join(dir, `test_image_processing_${model_type}.js`);
  if (!fs.existsSync(file)) {
    continue;
  }

  const { default: tests } = await import(file);
  describe(model_type, tests);
}

const sum = (array) => Number(array.reduce((a, b) => a + b, array instanceof BigInt64Array ? 0n : 0));
const avg = (array) => sum(array) / array.length;

const MODELS = {
  florence2: "Xenova/tiny-random-Florence2ForConditionalGeneration",
  qwen2_vl: "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
  idefics3: "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration",
  paligemma: "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration",
};

describe("Processors", () => {
  describe("Audio processors", () => {
    let audio;
    beforeAll(async () => {
      const url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.npy";
      const buffer = await (await fetch(url)).arrayBuffer();
      audio = Float32Array.from(new Float64Array(buffer));
    });

    it(
      "WhisperFeatureExtractor",
      async () => {
        const processor = await AutoProcessor.from_pretrained("Xenova/whisper-tiny.en");
        const { input_features } = await processor(audio);
        const { dims, data } = input_features;
        expect(dims).toEqual([1, 80, 3000]);
        expect(avg(data)).toBeCloseTo(-0.2813588131551941);
        expect(data[0]).toBeCloseTo(0.33168578147888184);
        expect(data[1]).toBeCloseTo(0.30986475944519043);
        expect(data[81]).toBeCloseTo(0.10727232694625854);
        expect(data[3001]).toBeCloseTo(0.2555035352706909);
      },
      MAX_TEST_TIME,
    );

    it(
      "ASTFeatureExtractor",
      async () => {
        const processor = await AutoProcessor.from_pretrained("Xenova/ast-finetuned-audioset-10-10-0.4593");
        {
          // truncation
          const { input_values } = await processor(audio);
          expect(input_values.dims).toEqual([1, 1024, 128]);

          expect(avg(input_values.data)).toBeCloseTo(-0.04054912979309085);
          expect(input_values.data[0]).toBeCloseTo(-0.5662586092948914);
          expect(input_values.data[1]).toBeCloseTo(-1.0300861597061157);
          expect(input_values.data[129]).toBeCloseTo(-1.084834098815918);
          expect(input_values.data[1025]).toBeCloseTo(-1.1204065084457397);
        }
        {
          // padding
          const { input_values } = await processor(audio.slice(0, 1000));
          expect(input_values.dims).toEqual([1, 1024, 128]); // [1, 4, 128] -> (padded to) -> [1, 1024, 128]

          expect(avg(input_values.data)).toBeCloseTo(0.4647964835166931);
          expect(input_values.data[0]).toBeCloseTo(-0.5662586092948914);
          expect(input_values.data[1]).toBeCloseTo(-1.0300861597061157);
          expect(input_values.data[129]).toBeCloseTo(-1.084834098815918);

          // padded values
          expect(input_values.data[1025]).toBeCloseTo(0.46703237295150757);
          expect(input_values.data[2049]).toBeCloseTo(0.46703237295150757);
          expect(input_values.data[10000]).toBeCloseTo(0.46703237295150757);
        }
      },
      MAX_TEST_TIME,
    );

    it(
      "SeamlessM4TFeatureExtractor",
      async () => {
        const processor = await AutoProcessor.from_pretrained("Xenova/wav2vec2-bert-CV16-en");
        {
          // normal
          const { input_features, attention_mask } = await processor(audio);
          const { dims, data } = input_features;
          expect(dims).toEqual([1, 649, 160]);
          expect(attention_mask.dims).toEqual([1, 649]);

          expect(avg(data)).toBeCloseTo(-2.938903875815413e-8);
          expect(data[0]).toBeCloseTo(1.1939343214035034);
          expect(data[1]).toBeCloseTo(0.7874255180358887);
          expect(data[160]).toBeCloseTo(-0.712975025177002);
          expect(data[161]).toBeCloseTo(0.045802414417266846);
          expect(data.at(-1)).toBeCloseTo(-1.3328346014022827);

          expect(sum(attention_mask.data)).toEqual(649);
        }
        {
          // padding (pad_to_multiple_of=2)
          const { input_features, attention_mask } = await processor(audio.slice(0, 10000));
          const { dims, data } = input_features;

          // [1, 61, 80] -> [1, 62, 80] -> [1, 31, 160]
          expect(dims).toEqual([1, 31, 160]);
          expect(attention_mask.dims).toEqual([1, 31]);

          expect(avg(data)).toBeCloseTo(0.01612919569015503);
          expect(data[0]).toBeCloseTo(0.9657132029533386);
          expect(data[1]).toBeCloseTo(0.12912897765636444);
          expect(data[160]).toBeCloseTo(-1.2364212274551392);
          expect(data[161]).toBeCloseTo(-0.9703778028488159);
          expect(data.at(-1)).toBeCloseTo(1); // padding value

          expect(sum(attention_mask.data)).toEqual(30);
        }
      },
      MAX_TEST_TIME,
    );

    it(
      "ClapFeatureExtractor",
      async () => {
        const processor = await AutoProcessor.from_pretrained("Xenova/clap-htsat-unfused");
        {
          // truncation
          // Since truncation uses a random strategy, we override
          // Math.random to ensure that the test is deterministic
          const originalRandom = Math.random;
          Math.random = () => 0.5;

          let long_audio = new Float32Array(500000);
          long_audio.set(audio);
          long_audio.set(audio, long_audio.length - audio.length);

          const { input_features } = await processor(long_audio);
          const { dims, data } = input_features;
          expect(dims).toEqual([1, 1, 1001, 64]);

          expect(avg(data)).toBeCloseTo(-37.94569396972656);
          expect(data[0]).toBeCloseTo(-53.32647705078125);
          expect(data[1]).toBeCloseTo(-47.76755142211914);
          expect(data[65]).toBeCloseTo(-36.32261276245117);
          expect(data[1002]).toBeCloseTo(-28.0314884185791);
          expect(data[10000]).toBeCloseTo(-21.905902862548828);
          expect(data[60000]).toBeCloseTo(-14.877863883972168);
          expect(data[64062]).toBeCloseTo(-37.9784049987793);
          expect(data[64063]).toBeCloseTo(-37.73963928222656);

          // Reset Math.random
          Math.random = originalRandom;
        }
        {
          // padding
          const { input_features } = await processor(audio);
          const { data, dims } = input_features;
          expect(dims).toEqual([1, 1, 1001, 64]);

          expect(avg(data)).toBeCloseTo(-34.99049377441406);
          expect(data[0]).toBeCloseTo(-21.32573890686035);
          expect(data[1]).toBeCloseTo(-26.168411254882812);
          expect(data[65]).toBeCloseTo(-29.716018676757812);
          expect(data[1002]).toBeCloseTo(-32.16273498535156);
          expect(data[10000]).toBeCloseTo(-19.9283390045166);

          // padded values
          expect(data[60000]).toBeCloseTo(-100.0);
          expect(data[64062]).toBeCloseTo(-100.0);
          expect(data[64063]).toBeCloseTo(-100.0);
        }
      },
      MAX_TEST_TIME,
    );

    it(
      "WeSpeakerFeatureExtractor",
      async () => {
        const processor = await AutoProcessor.from_pretrained("onnx-community/wespeaker-voxceleb-resnet34-LM");
        {
          // default
          const audio = new Float32Array(16000).map((_, i) => Math.sin(i / 100));
          const { input_features } = await processor(audio);
          const { dims, data } = input_features;
          expect(dims).toEqual([1, 98, 80]);

          expect(avg(data)).toBeCloseTo(5.461731689138105e-8);
          expect(data[0]).toBeCloseTo(-0.19300270080566406);
          expect(data[1]).toBeCloseTo(-0.05825042724609375);
          expect(data[78]).toBeCloseTo(0.2683420181274414);
          expect(data[79]).toBeCloseTo(0.26250171661376953);
          expect(data[80]).toBeCloseTo(0.19062232971191406);
          expect(data.at(-2)).toBeCloseTo(-0.43694400787353516);
          expect(data.at(-1)).toBeCloseTo(-0.4266204833984375);
        }

        {
          // pad to `min_num_frames`
          const audio = new Float32Array(3).map((_, i) => Math.sin(i / 100));
          const { input_features } = await processor(audio);
          const { dims, data } = input_features;
          expect(dims).toEqual([1, 9, 80]);

          expect(avg(data)).toBeCloseTo(-0.0000010093053181966146);
          expect(data[0]).toBeCloseTo(20.761859893798828);
          expect(data[1]).toBeCloseTo(21.02924346923828);
          expect(data[78]).toBeCloseTo(19.083993911743164);
          expect(data[79]).toBeCloseTo(18.003454208374023);
          expect(data[80]).toBeCloseTo(-2.595233917236328);
          expect(data.at(-2)).toBeCloseTo(-2.385499954223633);
          expect(data.at(-1)).toBeCloseTo(-2.2504329681396484);
        }
      },
      MAX_TEST_TIME,
    );
  });

  describe("Other processors", () => {
    describe(
      "FlorenceProcessor",
      () => {
        /** @type {import('../src/transformers.js').Florence2Processor} */
        let processor;
        let images = {};

        beforeAll(async () => {
          processor = await AutoProcessor.from_pretrained(MODELS.florence2);
          images = {
            beetle: await load_cached_image("beetle"),
            book_cover: await load_cached_image("book_cover"),
          };
        });

        describe("Prompt construction", () => {
          it("Construct prompt", async () => {
            const text = "<OD>";
            const prompts = processor.construct_prompts(text);
            const target = ["Locate the objects with category name in the image."];
            expect(prompts).toEqual(target);
          });

          it("Construct prompts", async () => {
            const texts = ["<MORE_DETAILED_CAPTION>", "Locate the objects with category name in the image.", "<OPEN_VOCABULARY_DETECTION>cat"];
            const prompts = processor.construct_prompts(texts);
            const target = ["Describe with a paragraph what is shown in the image.", "Locate the objects with category name in the image.", "Locate cat in the image."];
            expect(prompts).toEqual(target);
          });
        });

        describe("Post-process generation", () => {
          const TESTS = [
            {
              task: "<CAPTION>",
              generated_text: "</s><s>A green car parked in front of a yellow building.</s>",
              target: { "<CAPTION>": "A green car parked in front of a yellow building." },
              image: "beetle",
            },
            {
              task: "<DETAILED_CAPTION>",
              generated_text: "</s><s>The image shows a green Volkswagen Beetle parked in front of a yellow building with two brown doors. The sky is a mix of blue and white, and there are a few green trees in the background.</s>",
              target: { "<DETAILED_CAPTION>": "The image shows a green Volkswagen Beetle parked in front of a yellow building with two brown doors. The sky is a mix of blue and white, and there are a few green trees in the background." },
              image: "beetle",
            },
            {
              task: "<MORE_DETAILED_CAPTION>",
              generated_text: "</s><s>The image shows a vintage Volkswagen Beetle car parked on a cobblestone street in front of a yellow building with two wooden doors. The car is painted in a bright turquoise color and has a white stripe running along the side. It has two doors on either side of the car, one on top of the other, and a small window on the front. The building appears to be old and dilapidated, with peeling paint and crumbling walls. The sky is blue and there are trees in the background.</s>",
              target: { "<MORE_DETAILED_CAPTION>": "The image shows a vintage Volkswagen Beetle car parked on a cobblestone street in front of a yellow building with two wooden doors. The car is painted in a bright turquoise color and has a white stripe running along the side. It has two doors on either side of the car, one on top of the other, and a small window on the front. The building appears to be old and dilapidated, with peeling paint and crumbling walls. The sky is blue and there are trees in the background." },
              image: "beetle",
            },
            {
              task: "<OD>",
              generated_text: "</s><s><s><s>car<loc_53><loc_333><loc_933><loc_774>door<loc_712><loc_203><loc_906><loc_545>wheel<loc_704><loc_576><loc_866><loc_772><loc_149><loc_584><loc_310><loc_773></s>",
              target: {
                "<OD>": {
                  bboxes: [
                    [34.24, 160.08, 597.44, 371.76],
                    [456.0, 97.68, 580.16, 261.84],
                    [450.88, 276.72, 554.56, 370.8],
                    [95.68, 280.56, 198.72, 371.28],
                  ],
                  labels: ["car", "door", "wheel", "wheel"],
                },
              },
              image: "beetle",
            },
            {
              task: "<DENSE_REGION_CAPTION>",
              generated_text: "</s><s>turquoise Volkswagen Beetle<loc_52><loc_333><loc_932><loc_774>wheel<loc_704><loc_576><loc_864><loc_772><loc_148><loc_584><loc_308><loc_773></s>",
              target: {
                "<DENSE_REGION_CAPTION>": {
                  bboxes: [
                    [33.6, 160.08, 596.8, 371.76],
                    [450.88, 276.72, 553.28, 370.8],
                    [95.04, 280.56, 197.44, 371.28],
                  ],
                  labels: ["turquoise Volkswagen Beetle", "wheel", "wheel"],
                },
              },
              image: "beetle",
            },
            {
              task: "<REGION_PROPOSAL>",
              generated_text: "</s><s><s><s><loc_52><loc_333><loc_932><loc_774><loc_711><loc_203><loc_905><loc_545><loc_704><loc_576><loc_864><loc_772><loc_148><loc_584><loc_309><loc_773><loc_354><loc_184><loc_519><loc_342><loc_102><loc_555><loc_135><loc_616><loc_424><loc_503><loc_472><loc_514><loc_637><loc_642><loc_646><loc_668></s>",
              target: {
                "<REGION_PROPOSAL>": {
                  bboxes: [
                    [33.6, 160.08, 596.8, 371.76],
                    [455.36, 97.68, 579.52, 261.84],
                    [450.88, 276.72, 553.28, 370.8],
                    [95.04, 280.56, 198.08, 371.28],
                    [226.88, 88.56, 332.48, 164.4],
                    [65.6, 266.64, 86.72, 295.92],
                    [271.68, 241.68, 302.4, 246.96],
                    [408.0, 308.4, 413.76, 320.88],
                  ],
                  labels: ["", "", "", "", "", "", "", ""],
                },
              },
              image: "beetle",
            },
            {
              task: "<CAPTION_TO_PHRASE_GROUNDING>",
              text_input: "A green car parked in front of a yellow building.",
              generated_text: "</s><s><s><s>A green car<loc_54><loc_330><loc_911><loc_780>a yellow building<loc_0><loc_8><loc_998><loc_635></s>",
              target: {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                  bboxes: [
                    [34.88, 158.64, 583.36, 374.64],
                    [0.32, 4.08, 639.04, 305.04],
                  ],
                  labels: ["A green car", "a yellow building"],
                },
              },
              image: "beetle",
            },
            // {
            //     task: "<REFERRING_EXPRESSION_SEGMENTATION>",
            //     text_input: "a green car",
            //     generated_text: "</s><s><s><s><loc_279><loc_378><loc_282><loc_376><loc_285><loc_376><loc_293><loc_370><loc_296><loc_370><loc_301><loc_366><loc_304><loc_366><loc_309><loc_362><loc_313><loc_360><loc_318><loc_358><loc_323><loc_355><loc_327><loc_353><loc_334><loc_351><loc_340><loc_349><loc_346><loc_347><loc_353><loc_345><loc_360><loc_343><loc_370><loc_341><loc_381><loc_339><loc_395><loc_337><loc_414><loc_335><loc_486><loc_335><loc_514><loc_337><loc_528><loc_339><loc_539><loc_341><loc_547><loc_343><loc_553><loc_345><loc_560><loc_347><loc_566><loc_349><loc_572><loc_351><loc_578><loc_353><loc_583><loc_355><loc_586><loc_358><loc_589><loc_362><loc_592><loc_368><loc_594><loc_374><loc_597><loc_378><loc_600><loc_385><loc_603><loc_391><loc_605><loc_397><loc_608><loc_401><loc_609><loc_408><loc_612><loc_414><loc_616><loc_420><loc_619><loc_426><loc_622><loc_433><loc_630><loc_443><loc_634><loc_445><loc_639><loc_451><loc_644><loc_458><loc_674><loc_458><loc_675><loc_460><loc_691><loc_462><loc_713><loc_462><loc_727><loc_464><loc_738><loc_466><loc_747><loc_468><loc_757><loc_470><loc_765><loc_472><loc_771><loc_474><loc_777><loc_476><loc_783><loc_478><loc_788><loc_481><loc_793><loc_483><loc_797><loc_485><loc_802><loc_487><loc_807><loc_491><loc_810><loc_491><loc_818><loc_497><loc_821><loc_497><loc_824><loc_499><loc_827><loc_503><loc_832><loc_505><loc_837><loc_510><loc_841><loc_516><loc_846><loc_520><loc_852><loc_524><loc_857><loc_526><loc_860><loc_526><loc_865><loc_528><loc_869><loc_532><loc_872><loc_532><loc_882><loc_539><loc_885><loc_543><loc_888><loc_543><loc_891><loc_545><loc_894><loc_549><loc_896><loc_553><loc_897><loc_559><loc_897><loc_566><loc_896><loc_568><loc_894><loc_574><loc_894><loc_582><loc_896><loc_595><loc_897><loc_597><loc_899><loc_603><loc_900><loc_609><loc_902><loc_622><loc_902><loc_628><loc_900><loc_630><loc_899><loc_647><loc_899><loc_651><loc_900><loc_653><loc_902><loc_659><loc_902><loc_668><loc_897><loc_670><loc_888><loc_672><loc_874><loc_672><loc_865><loc_674><loc_863><loc_693><loc_862><loc_701><loc_860><loc_707><loc_859><loc_714><loc_857><loc_718><loc_854><loc_722><loc_852><loc_728><loc_849><loc_734><loc_846><loc_741><loc_835><loc_755><loc_830><loc_759><loc_821><loc_766><loc_816><loc_768><loc_810><loc_770><loc_774><loc_770><loc_765><loc_768><loc_760><loc_766><loc_755><loc_764><loc_749><loc_759><loc_744><loc_755><loc_738><loc_749><loc_727><loc_734><loc_724><loc_728><loc_721><loc_722><loc_719><loc_718><loc_719><loc_714><loc_716><loc_707><loc_715><loc_701><loc_715><loc_697><loc_713><loc_693><loc_710><loc_689><loc_707><loc_691><loc_700><loc_701><loc_697><loc_703><loc_666><loc_701><loc_663><loc_701><loc_661><loc_703><loc_657><loc_705><loc_647><loc_707><loc_644><loc_707><loc_642><loc_705><loc_594><loc_703><loc_339><loc_703><loc_337><loc_705><loc_329><loc_707><loc_323><loc_707><loc_318><loc_705><loc_315><loc_703><loc_312><loc_699><loc_309><loc_697><loc_304><loc_697><loc_301><loc_701><loc_299><loc_705><loc_299><loc_709><loc_298><loc_714><loc_295><loc_718><loc_293><loc_724><loc_290><loc_728><loc_288><loc_734><loc_285><loc_741><loc_276><loc_753><loc_271><loc_757><loc_266><loc_761><loc_260><loc_766><loc_255><loc_768><loc_251><loc_770><loc_240><loc_772><loc_205><loc_772><loc_199><loc_770><loc_194><loc_768><loc_185><loc_761><loc_180><loc_757><loc_174><loc_751><loc_166><loc_741><loc_163><loc_734><loc_161><loc_728><loc_158><loc_724><loc_157><loc_720><loc_155><loc_714><loc_155><loc_707><loc_154><loc_703><loc_149><loc_697><loc_146><loc_695><loc_135><loc_695><loc_125><loc_697><loc_124><loc_699><loc_116><loc_701><loc_103><loc_701><loc_99><loc_697><loc_83><loc_697><loc_78><loc_695><loc_75><loc_691><loc_75><loc_684><loc_78><loc_680><loc_80><loc_676><loc_80><loc_672><loc_69><loc_670><loc_63><loc_668><loc_60><loc_666><loc_58><loc_661><loc_56><loc_653><loc_56><loc_639><loc_60><loc_634><loc_66><loc_632><loc_72><loc_630><loc_86><loc_628><loc_102><loc_628><loc_105><loc_626><loc_108><loc_622><loc_110><loc_618><loc_110><loc_609><loc_108><loc_607><loc_107><loc_601><loc_105><loc_593><loc_105><loc_576><loc_107><loc_570><loc_108><loc_566><loc_113><loc_559><loc_116><loc_557><loc_121><loc_555><loc_124><loc_555><loc_127><loc_551><loc_125><loc_543><loc_127><loc_539><loc_130><loc_534><loc_138><loc_534><loc_141><loc_532><loc_144><loc_528><loc_144><loc_526><loc_152><loc_514><loc_179><loc_478><loc_183><loc_472><loc_191><loc_464><loc_196><loc_460><loc_197><loc_460><loc_202><loc_456><loc_208><loc_449><loc_216><loc_441><loc_224><loc_433><loc_233><loc_420><loc_240><loc_414><loc_241><loc_414><loc_246><loc_410><loc_254><loc_401><loc_263><loc_389><loc_268><loc_385><loc_276><loc_381><loc_279><loc_376></s>",
            //     target: {
            //         '<REFERRING_EXPRESSION_SEGMENTATION>': {
            //             polygons: [[[[178.88, 181.68, 180.8, 180.72, 182.72, 180.72, 187.84, 177.84, 189.76, 177.84, 192.96, 175.92, 194.88, 175.92, 198.08, 174, 200.64, 173.04, 203.84, 172.08, 207.04, 170.64, 209.6, 169.68, 214.08, 168.72, 217.92, 167.76, 221.76, 166.8, 226.24, 165.84, 230.72, 164.88, 237.12, 163.92, 244.16, 162.96, 253.12, 162, 265.28, 161.04, 311.36, 161.04, 329.28, 162, 338.24, 162.96, 345.28, 163.92, 350.4, 164.88, 354.24, 165.84, 358.72, 166.8, 362.56, 167.76, 366.4, 168.72, 370.24, 169.68, 373.44, 170.64, 375.36, 172.08, 377.28, 174, 379.2, 176.88, 380.48, 179.76, 382.4, 181.68, 384.32, 185.04, 386.24, 187.92, 387.52, 190.8, 389.44, 192.72, 390.08, 196.08, 392, 198.96, 394.56, 201.84, 396.48, 204.72, 398.4, 208.08, 403.52, 212.88, 406.08, 213.84, 409.28, 216.72, 412.48, 220.08, 431.68, 220.08, 432.32, 221.04, 442.56, 222, 456.64, 222, 465.6, 222.96, 472.64, 223.92, 478.4, 224.88, 484.8, 225.84, 489.92, 226.8, 493.76, 227.76, 497.6, 228.72, 501.44, 229.68, 504.64, 231.12, 507.84, 232.08, 510.4, 233.04, 513.6, 234, 516.8, 235.92, 518.72, 235.92, 523.84, 238.8, 525.76, 238.8, 527.68, 239.76, 529.6, 241.68, 532.8, 242.64, 536, 245.04, 538.56, 247.92, 541.76, 249.84, 545.6, 251.76, 548.8, 252.72, 550.72, 252.72, 553.92, 253.68, 556.48, 255.6, 558.4, 255.6, 564.8, 258.96, 566.72, 260.88, 568.64, 260.88, 570.56, 261.84, 572.48, 263.76, 573.76, 265.68, 574.4, 268.56, 574.4, 271.92, 573.76, 272.88, 572.48, 275.76, 572.48, 279.6, 573.76, 285.84, 574.4, 286.8, 575.68, 289.68, 576.32, 292.56, 577.6, 298.8, 577.6, 301.68, 576.32, 302.64, 575.68, 310.8, 575.68, 312.72, 576.32, 313.68, 577.6, 316.56, 577.6, 320.88, 574.4, 321.84, 568.64, 322.8, 559.68, 322.8, 553.92, 323.76, 552.64, 332.88, 552, 336.72, 550.72, 339.6, 550.08, 342.96, 548.8, 344.88, 546.88, 346.8, 545.6, 349.68, 543.68, 352.56, 541.76, 355.92, 534.72, 362.64, 531.52, 364.56, 525.76, 367.92, 522.56, 368.88, 518.72, 369.84, 495.68, 369.84, 489.92, 368.88, 486.72, 367.92, 483.52, 366.96, 479.68, 364.56, 476.48, 362.64, 472.64, 359.76, 465.6, 352.56, 463.68, 349.68, 461.76, 346.8, 460.48, 344.88, 460.48, 342.96, 458.56, 339.6, 457.92, 336.72, 457.92, 334.8, 456.64, 332.88, 454.72, 330.96, 452.8, 331.92, 448.32, 336.72, 446.4, 337.68, 426.56, 336.72, 424.64, 336.72, 423.36, 337.68, 420.8, 338.64, 414.4, 339.6, 412.48, 339.6, 411.2, 338.64, 380.48, 337.68, 217.28, 337.68, 216, 338.64, 210.88, 339.6, 207.04, 339.6, 203.84, 338.64, 201.92, 337.68, 200, 335.76, 198.08, 334.8, 194.88, 334.8, 192.96, 336.72, 191.68, 338.64, 191.68, 340.56, 191.04, 342.96, 189.12, 344.88, 187.84, 347.76, 185.92, 349.68, 184.64, 352.56, 182.72, 355.92, 176.96, 361.68, 173.76, 363.6, 170.56, 365.52, 166.72, 367.92, 163.52, 368.88, 160.96, 369.84, 153.92, 370.8, 131.52, 370.8, 127.68, 369.84, 124.48, 368.88, 118.72, 365.52, 115.52, 363.6, 111.68, 360.72, 106.56, 355.92, 104.64, 352.56, 103.36, 349.68, 101.44, 347.76, 100.8, 345.84, 99.52, 342.96, 99.52, 339.6, 98.88, 337.68, 95.68, 334.8, 93.76, 333.84, 86.72, 333.84, 80.32, 334.8, 79.68, 335.76, 74.56, 336.72, 66.24, 336.72, 63.68, 334.8, 53.44, 334.8, 50.24, 333.84, 48.32, 331.92, 48.32, 328.56, 50.24, 326.64, 51.52, 324.72, 51.52, 322.8, 44.48, 321.84, 40.64, 320.88, 38.72, 319.92, 37.44, 317.52, 36.16, 313.68, 36.16, 306.96, 38.72, 304.56, 42.56, 303.6, 46.4, 302.64, 55.36, 301.68, 65.6, 301.68, 67.52, 300.72, 69.44, 298.8, 70.72, 296.88, 70.72, 292.56, 69.44, 291.6, 68.8, 288.72, 67.52, 284.88, 67.52, 276.72, 68.8, 273.84, 69.44, 271.92, 72.64, 268.56, 74.56, 267.6, 77.76, 266.64, 79.68, 266.64, 81.6, 264.72, 80.32, 260.88, 81.6, 258.96, 83.52, 256.56, 88.64, 256.56, 90.56, 255.6, 92.48, 253.68, 92.48, 252.72, 97.6, 246.96, 114.88, 229.68, 117.44, 226.8, 122.56, 222.96, 125.76, 221.04, 126.4, 221.04, 129.6, 219.12, 133.44, 215.76, 138.56, 211.92, 143.68, 208.08, 149.44, 201.84, 153.92, 198.96, 154.56, 198.96, 157.76, 197.04, 162.88, 192.72, 168.64, 186.96, 171.84, 185.04, 176.96, 183.12, 178.88, 180.72]]]],
            //             labels: [''],
            //         }
            //     },
            //     image: 'beetle',
            // },
            // {
            //     task: "<REGION_TO_SEGMENTATION>",
            //     text_input: "<loc_702><loc_575><loc_866><loc_772>",
            //     generated_text: "</s><s><s><s><loc_734><loc_600><loc_740><loc_594><loc_745><loc_590><loc_748><loc_588><loc_751><loc_588><loc_756><loc_584><loc_760><loc_582><loc_765><loc_580><loc_773><loc_578><loc_800><loc_578><loc_804><loc_580><loc_809><loc_582><loc_814><loc_586><loc_817><loc_586><loc_820><loc_590><loc_825><loc_592><loc_829><loc_596><loc_834><loc_600><loc_848><loc_619><loc_851><loc_625><loc_854><loc_631><loc_859><loc_644><loc_861><loc_650><loc_862><loc_656><loc_864><loc_665><loc_864><loc_692><loc_862><loc_702><loc_861><loc_708><loc_859><loc_715><loc_856><loc_723><loc_853><loc_729><loc_850><loc_735><loc_845><loc_744><loc_839><loc_752><loc_831><loc_760><loc_826><loc_764><loc_823><loc_766><loc_818><loc_768><loc_814><loc_770><loc_806><loc_773><loc_782><loc_773><loc_768><loc_770><loc_762><loc_768><loc_757><loc_766><loc_748><loc_760><loc_743><loc_756><loc_737><loc_750><loc_726><loc_735><loc_723><loc_729><loc_720><loc_723><loc_718><loc_719><loc_718><loc_715><loc_715><loc_708><loc_713><loc_702><loc_712><loc_696><loc_710><loc_688><loc_710><loc_658><loc_712><loc_648><loc_713><loc_640><loc_715><loc_633><loc_718><loc_627><loc_718><loc_623><loc_720><loc_619><loc_723><loc_613></s>",
            //     target: {
            //         '<REGION_TO_SEGMENTATION>': {
            //             polygons: [[[[470.08, 288.24, 473.92, 285.36, 477.12, 283.44, 479.04, 282.48, 480.96, 282.48, 484.16, 280.56, 486.72, 279.6, 489.92, 278.64, 495.04, 277.68, 512.32, 277.68, 514.88, 278.64, 518.08, 279.6, 521.28, 281.52, 523.2, 281.52, 525.12, 283.44, 528.32, 284.4, 530.88, 286.32, 534.08, 288.24, 543.04, 297.36, 544.96, 300.24, 546.88, 303.12, 550.08, 309.36, 551.36, 312.24, 552, 315.12, 553.28, 319.44, 553.28, 332.4, 552, 337.2, 551.36, 340.08, 550.08, 343.44, 548.16, 347.28, 546.24, 350.16, 544.32, 353.04, 541.12, 357.36, 537.28, 361.2, 532.16, 365.04, 528.96, 366.96, 527.04, 367.92, 523.84, 368.88, 521.28, 369.84, 516.16, 371.28, 500.8, 371.28, 491.84, 369.84, 488, 368.88, 484.8, 367.92, 479.04, 365.04, 475.84, 363.12, 472, 360.24, 464.96, 353.04, 463.04, 350.16, 461.12, 347.28, 459.84, 345.36, 459.84, 343.44, 457.92, 340.08, 456.64, 337.2, 456, 334.32, 454.72, 330.48, 454.72, 316.08, 456, 311.28, 456.64, 307.44, 457.92, 304.08, 459.84, 301.2, 459.84, 299.28, 461.12, 297.36, 463.04, 294.48]]]],
            //             labels: [''],
            //         }
            //     },
            //     image: 'beetle',
            // },
            // {
            //     task: "<OPEN_VOCABULARY_DETECTION>",
            //     text_input: "a green car",
            //     generated_text: "</s><s><s>a green car<loc_53><loc_330><loc_910><loc_779></s>",
            //     target: {
            //         '<OPEN_VOCABULARY_DETECTION>': {
            //             bboxes: [[34.24, 158.64, 582.72, 374.16]],
            //             bboxes_labels: ['a green car'],
            //             polygons: [],
            //             polygons_labels: [],
            //         }
            //     },
            //     image: 'beetle',
            // },
            {
              task: "<REGION_TO_CATEGORY>",
              text_input: "<loc_52><loc_332><loc_932><loc_774>",
              generated_text: "</s><s>car<loc_52><loc_332><loc_932><loc_774></s>",
              target: { "<REGION_TO_CATEGORY>": "car<loc_52><loc_332><loc_932><loc_774>" },
              image: "beetle",
            },
            {
              task: "<REGION_TO_DESCRIPTION>",
              text_input: "<loc_52><loc_332><loc_932><loc_774>",
              generated_text: "</s><s>turquoise Volkswagen Beetle<loc_52><loc_332><loc_932><loc_774></s>",
              target: { "<REGION_TO_DESCRIPTION>": "turquoise Volkswagen Beetle<loc_52><loc_332><loc_932><loc_774>" },
              image: "beetle",
            },
            {
              task: "<OCR>",
              generated_text: "</s><s>CUDAFOR ENGINEERSAn Introduction to High-PerformanceParallel ComputingDUANE STORTIMETE YURTOGLU</s>",
              target: { "<OCR>": "CUDAFOR ENGINEERSAn Introduction to High-PerformanceParallel ComputingDUANE STORTIMETE YURTOGLU" },
              image: "book_cover",
            },
            {
              task: "<OCR_WITH_REGION>",
              generated_text: "</s><s><s><s>CUDA<loc_414><loc_100><loc_932><loc_100><loc_932><loc_229><loc_414><loc_229>FOR ENGINEERS<loc_359><loc_241><loc_932><loc_241><loc_932><loc_298><loc_359><loc_298>An Introduction to High-Performance<loc_287><loc_330><loc_934><loc_332><loc_934><loc_368><loc_287><loc_366>Parallel Computing<loc_595><loc_368><loc_934><loc_372><loc_934><loc_408><loc_595><loc_404>DUANE STORTI<loc_660><loc_882><loc_934><loc_882><loc_934><loc_912><loc_660><loc_912>METE YURTOGLU<loc_625><loc_920><loc_934><loc_920><loc_934><loc_950><loc_625><loc_950></s>",
              target: {
                "<OCR_WITH_REGION>": {
                  quad_boxes: [
                    [167.0435028076172, 50.25, 375.7974853515625, 50.25, 375.7974853515625, 114.75, 167.0435028076172, 114.75],
                    [144.8784942626953, 120.75, 375.7974853515625, 120.75, 375.7974853515625, 149.25, 144.8784942626953, 149.25],
                    [115.86249542236328, 165.25, 376.6034851074219, 166.25, 376.6034851074219, 184.25, 115.86249542236328, 183.25],
                    [239.9864959716797, 184.25, 376.6034851074219, 186.25, 376.6034851074219, 204.25, 239.9864959716797, 202.25],
                    [266.1814880371094, 441.25, 376.6034851074219, 441.25, 376.6034851074219, 456.25, 266.1814880371094, 456.25],
                    [252.0764923095703, 460.25, 376.6034851074219, 460.25, 376.6034851074219, 475.25, 252.0764923095703, 475.25],
                  ],

                  // NOTE: Python version has a bug here, it should be "CUDA" instead of "</s>CUDA"
                  labels: [/* '</s>CUDA' */ "CUDA", "FOR ENGINEERS", "An Introduction to High-Performance", "Parallel Computing", "DUANE STORTI", "METE YURTOGLU"],
                },
              },
              image: "book_cover",
            },
          ];

          for (const { task, generated_text, target, image } of TESTS) {
            it(task, () => {
              const result = processor.post_process_generation(generated_text, task, images[image].size);
              expect(result).toBeCloseToNested(target, 4);
            });
          }
        });
      },
      MAX_TEST_TIME,
    );

    describe(
      "Qwen2VLProcessor",
      () => {
        /** @type {import('../src/transformers.js').Qwen2VLProcessor} */
        let processor;
        let images = {};

        beforeAll(async () => {
          processor = await AutoProcessor.from_pretrained(MODELS.qwen2_vl);
          images = {
            white_image: await load_cached_image("white_image"),
          };
        });

        it("Image and text", async () => {
          const conversation = [
            {
              role: "user",
              content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
            },
          ];

          const text = processor.apply_chat_template(conversation, {
            add_generation_prompt: true,
          });
          const { input_ids, attention_mask, pixel_values, image_grid_thw } = await processor(text, images.white_image);

          expect(input_ids.dims).toEqual([1, 89]);
          expect(attention_mask.dims).toEqual([1, 89]);
          expect(pixel_values.dims).toEqual([256, 1176]);
          expect(image_grid_thw.dims).toEqual([1, 3]);
        });
      },
      MAX_TEST_TIME,
    );

    describe(
      "PaliGemmaProcessor",
      () => {
        /** @type {import('../src/transformers.js').PaliGemmaProcessor} */
        let processor;
        let images = {};

        beforeAll(async () => {
          processor = await AutoProcessor.from_pretrained(MODELS.paligemma);
          images = {
            white_image: await load_cached_image("white_image"),
          };
        });

        it("Image-only (default text)", async () => {
          const { input_ids, pixel_values } = await processor(images.white_image);
          expect(input_ids.dims).toEqual([1, 258]);
          expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        });

        it("Single image & text", async () => {
          const { input_ids, pixel_values } = await processor(images.white_image, "<image>What is on the flower?");
          expect(input_ids.dims).toEqual([1, 264]);
          expect(pixel_values.dims).toEqual([1, 3, 224, 224]);
        });

        it("Multiple images & text", async () => {
          const { input_ids, pixel_values } = await processor([images.white_image, images.white_image], "<image><image>Describe the images.");
          expect(input_ids.dims).toEqual([1, 518]);
          expect(pixel_values.dims).toEqual([2, 3, 224, 224]);
        });
      },
      MAX_TEST_TIME,
    );
  });
});
