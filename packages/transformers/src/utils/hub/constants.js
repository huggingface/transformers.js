export const ERROR_MAPPING = {
    // 4xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#client_error_responses)
    400: 'Bad request error occurred while trying to load file',
    401: 'Unauthorized access to file',
    403: 'Forbidden access to file',
    404: 'Could not locate file',
    408: 'Request timeout error occurred while trying to load file',

    // 5xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#server_error_responses)
    500: 'Internal server error error occurred while trying to load file',
    502: 'Bad gateway error occurred while trying to load file',
    503: 'Service unavailable error occurred while trying to load file',
    504: 'Gateway timeout error occurred while trying to load file',
};

export const MAX_EXTERNAL_DATA_CHUNKS = 100;

export const REPO_ID_REGEX = /^(\b[\w\-.]+\b\/)?\b[\w\-.]{1,96}\b$/;
