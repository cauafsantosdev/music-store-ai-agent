"""Strict LangChain adapters delegating to existing deterministic store operations."""

import json
from collections.abc import Callable
from functools import partial
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.agent.demo import verified_snapshot
from music_store_agent.agent.errors import PUBLIC_MESSAGES, AgentFailure, ErrorCode
from music_store_agent.policies.errors import PolicyIndexMissing, PolicySearchError
from music_store_agent.services.order_verification import (
    OrderLookupError,
    normalize_customer_email,
)
from music_store_agent.tools import orders, policies, products

type PaymentInput = Literal["pix", "credit_card", "debit_card", "boleto"] | None
type ToolResult = BaseModel | tuple[BaseModel, ...] | None


class ToolArguments(BaseModel):
    """Reject unknown fields and implicit coercions before business tool execution."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CatalogArguments(ToolArguments):
    """Bound catalog inputs while requiring exact decimal-string budget values."""

    query: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=100)
    max_price_brl: str | None = Field(default=None, max_length=30)
    min_price_brl: str | None = Field(default=None, max_length=30)
    payment_method: PaymentInput = None
    limit: int = Field(default=5, ge=1, le=10)


class ProductArguments(ToolArguments):
    """Select an exact source product and optional explicit payment context."""

    product_id: int = Field(ge=1, le=2_147_483_647)
    payment_method: PaymentInput = None


class PromotionArguments(ToolArguments):
    """Limit promotion results to a small customer-facing selection."""

    limit: int = Field(default=5, ge=1, le=10)


class OrderArguments(ToolArguments):
    """Require an order ID and a reserved synthetic email for demonstration use."""

    order_id: int = Field(ge=1, le=2_147_483_647)
    customer_email: str = Field(min_length=3, max_length=254)

    @field_validator("customer_email")
    @classmethod
    def validate_synthetic_email(cls, value: str) -> str:
        """Reject real email domains before accessing tracking records.

        Args:
            value: Model-supplied synthetic association address.

        Returns:
            Validated, trimmed example.test address.

        Raises:
            ValueError: If the address is malformed or outside the demo namespace.
        """
        email = normalize_customer_email(value)
        if email.rsplit("@", 1)[-1].casefold() != "example.test":
            raise ValueError("Only synthetic example.test addresses are permitted.")
        return email


class PolicyArguments(ToolArguments):
    """Bound source-retrieval input and the number of policy excerpts."""

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=3)


def tool_failure(code: ErrorCode) -> dict[str, object]:
    """Create a safe structured tool failure without diagnostic or argument values.

    Args:
        code: Stable application failure category.

    Returns:
        A JSON-compatible error payload in Brazilian Portuguese.
    """
    return {"status": "error", "code": code.value, "message": PUBLIC_MESSAGES[code]}


def invalid_arguments(error: ValidationError) -> str:
    """Replace Pydantic diagnostics that might otherwise echo model-supplied PII.

    Args:
        error: Validation details, deliberately not serialized or logged.

    Returns:
        A safe JSON tool-error message.
    """
    return json.dumps(tool_failure(ErrorCode.INVALID_ARGUMENTS), ensure_ascii=False)


def _execute(
    engine: Engine,
    operation: Callable[[Engine], ToolResult],
    *,
    missing: ErrorCode = ErrorCode.NOT_FOUND,
) -> dict[str, object]:
    """Verify provenance and execute a real operation inside the same snapshot.

    Args:
        engine: Explicit synthetic schema engine.
        operation: Existing deterministic tool with its public inputs bound.
        missing: Uniform failure code when the existing tool returns None.

    Returns:
        Compact JSON-compatible data, or a sanitized classified failure.
    """
    try:
        with verified_snapshot(engine) as bind:
            result = operation(bind)
        if result is None:
            return tool_failure(missing)
        if isinstance(result, tuple):
            if not result:
                return {
                    "status": "empty",
                    "code": ErrorCode.NOT_FOUND.value,
                    "data": [],
                }
            data = [item.model_dump(mode="json") for item in result]
        else:
            data = result.model_dump(mode="json")
        return {"status": "ok", "data": data}
    except AgentFailure as error:
        return tool_failure(error.code)
    except PolicyIndexMissing:
        return tool_failure(ErrorCode.MISSING_POLICY)
    except (SQLAlchemyError, OrderLookupError, PolicySearchError):
        return tool_failure(ErrorCode.DATABASE)
    except ValueError:
        return tool_failure(ErrorCode.INVALID_ARGUMENTS)


def build_tools(engine: Engine) -> tuple[BaseTool, ...]:
    """Register six schema-safe adapters without exposing engine/session dependencies.

    Args:
        engine: Explicit engine checked again within every tool invocation.

    Returns:
        Structured LangChain tools retaining existing business rules and result fields.
    """

    def search_products(
        query: str | None = None,
        category: str | None = None,
        max_price_brl: str | None = None,
        min_price_brl: str | None = None,
        payment_method: PaymentInput = None,
        limit: int = 5,
    ) -> dict[str, object]:
        """Search available instruments using exact decimal-string budgets.

        Args:
            query: Literal product or category text.
            category: Optional category substring.
            max_price_brl: Inclusive upper price, supplied as a decimal string.
            min_price_brl: Inclusive lower price, supplied as a decimal string.
            payment_method: Explicit payment context; omitted means no assumed PIX.
            limit: Maximum results from 1 through 10.

        Returns:
            Real catalog records with evaluated availability and price-conflict flags.
        """
        return _execute(
            engine,
            partial(
                products.search_products,
                query=query,
                category=category,
                max_price_brl=max_price_brl,
                min_price_brl=min_price_brl,
                payment_method=payment_method,
                limit=limit,
            ),
        )

    def get_product_details(
        product_id: int, payment_method: PaymentInput = None
    ) -> dict[str, object]:
        """Retrieve source specifications and explicit availability for one product.

        Args:
            product_id: Exact source product identifier.
            payment_method: Optional explicit payment context.

        Returns:
            Source facts and eligibility flags, including unavailable products.
        """
        return _execute(
            engine,
            partial(
                products.get_product_details,
                product_id=product_id,
                payment_method=payment_method,
            ),
        )

    def get_product_pricing(
        product_id: int, payment_method: PaymentInput = None
    ) -> dict[str, object]:
        """Obtain verified prices without stacking PIX and active promotions.

        Args:
            product_id: Exact source product identifier.
            payment_method: Optional explicit PIX or other accepted payment method.

        Returns:
            Exact monetary strings, promotions, or structured price conflicts.
        """
        return _execute(
            engine,
            partial(
                products.get_product_pricing,
                product_id=product_id,
                payment_method=payment_method,
            ),
        )

    def get_active_promotions(limit: int = 5) -> dict[str, object]:
        """Inspect active promotions while preserving availability and conflict flags.

        Args:
            limit: Maximum products from 1 through 10.

        Returns:
            Source promotion records with verified-offer decisions.
        """
        return _execute(engine, partial(products.get_active_promotions, limit=limit))

    def get_order_status(order_id: int, customer_email: str) -> dict[str, object]:
        """Check a synthetic order association using both order number and email.

        Args:
            order_id: Exact source order identifier.
            customer_email: Synthetic example.test email associated with that order.

        Returns:
            Only approved tracking fields, or one uniform unverified outcome.
        """
        return _execute(
            engine,
            partial(
                orders.get_order_status,
                order_id=order_id,
                customer_email=customer_email,
            ),
            missing=ErrorCode.ORDER_UNVERIFIED,
        )

    def search_store_policies(query: str, limit: int = 3) -> dict[str, object]:
        """Retrieve original Portuguese policy passages with section/page references.

        Args:
            query: Concise policy question in Portuguese or supported English concepts.
            limit: Maximum excerpts from 1 through 3.

        Returns:
            Bounded source references, never model-generated policy answers.
        """
        return _execute(
            engine, partial(policies.search_store_policies, query=query, limit=limit)
        )

    registrations = (
        (search_products, CatalogArguments),
        (get_product_details, ProductArguments),
        (get_product_pricing, ProductArguments),
        (get_active_promotions, PromotionArguments),
        (get_order_status, OrderArguments),
        (search_store_policies, PolicyArguments),
    )
    return tuple(
        StructuredTool.from_function(
            function,
            args_schema=schema,
            handle_validation_error=invalid_arguments,
        )
        for function, schema in registrations
    )
