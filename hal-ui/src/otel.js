import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { resourceFromAttributes } from '@opentelemetry/resources';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION } from '@opentelemetry/semantic-conventions';
import { context, propagation, trace, ROOT_CONTEXT } from '@opentelemetry/api';

const spanMap = new Map();

const processor = new BatchSpanProcessor(
    new OTLPTraceExporter({ url: 'http://localhost:4318/v1/traces' })
)

const provider = new WebTracerProvider({
    resource: resourceFromAttributes({
        [ATTR_SERVICE_NAME]: 'hal',
        [ATTR_SERVICE_VERSION]: '0.1.0',
        'environment': 'dev',
    }),
    spanProcessors: [processor],
});
provider.register();


export function createRootSpan(name) {
    const tracer = provider.getTracer('hal-ui');
    const span = tracer.startSpan(`client:${name}`, {}, ROOT_CONTEXT);
    const spanContext = trace.setSpan(context.active(), span);
    const tracePropagation = {};
    propagation.inject(spanContext, tracePropagation);
    console.log('tracePropagation:', tracePropagation);
    spanMap.set(span.spanContext().spanId, span);
    return { span, traceparent: tracePropagation.traceparent };
}

export function fetchSpan(spanId) {
    return spanMap.get(spanId);
}

export function endRootSpan(spanId) {
    const span = fetchSpan(spanId);
    span.end();
    spanMap.delete(spanId);
}
