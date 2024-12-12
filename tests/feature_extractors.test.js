import { init } from "./init.js";
import { collect_and_execute_tests } from "./test_utils.js";

init();
await collect_and_execute_tests("Feature extractors", "feature_extraction");
