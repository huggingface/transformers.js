/**
 * Response parsing: convert model-emitted text into the assistant message used by chat templates,
 * driven by a declarative `response_template` spec.
 */
export { parse_response, ResponseParser } from './response_parser.js';
