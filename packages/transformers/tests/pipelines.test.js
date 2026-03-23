import { init } from "./init.js";
import { collect_and_execute_pipeline_tests } from "./test_utils.js";

// Initialise the testing environment
init();
await collect_and_execute_pipeline_tests("Pipelines");
