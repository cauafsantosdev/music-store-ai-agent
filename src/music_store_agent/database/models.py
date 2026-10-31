"""Typed operational records preserving the identifiers and values in source CSVs."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from music_store_agent.database.base import Base

type JsonValue = (
    dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None
)


class Category(Base):
    """A source catalog category and its associated products."""

    __tablename__: str = "categories"

    category_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    products: Mapped[list[Product]] = relationship(back_populates="category")


class Product(Base):
    """A catalog product with exact BRL price and structured specifications."""

    __tablename__: str = "products"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("price_brl >= 0", name="products_price_nonnegative"),
        CheckConstraint("stock_quantity >= 0", name="products_stock_nonnegative"),
    )

    product_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.category_id"), index=True
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    price_brl: Mapped[Decimal] = mapped_column(Numeric(14, 2, asdecimal=True))
    stock_quantity: Mapped[int] = mapped_column(Integer)
    # Status is source data, not an eligibility rule or a closed application enum.
    status: Mapped[str] = mapped_column(String)
    specs: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[date] = mapped_column(Date)
    category: Mapped[Category] = relationship(back_populates="products")
    order_items: Mapped[list[OrderItem]] = relationship(back_populates="product")
    promotions: Mapped[list[Promotion]] = relationship(back_populates="product")


class Customer(Base):
    """An independently identified customer; email addresses need not be unique."""

    __tablename__: str = "customers"

    customer_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(String)
    phone: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    city: Mapped[str] = mapped_column(String)
    orders: Mapped[list[Order]] = relationship(back_populates="customer")


class Order(Base):
    """A source order whose recorded total is retained without recalculation."""

    __tablename__: str = "orders"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("total_brl >= 0", name="orders_total_nonnegative"),
    )

    order_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_id"), index=True
    )
    order_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String)
    total_brl: Mapped[Decimal] = mapped_column(Numeric(14, 2, asdecimal=True))
    payment_method: Mapped[str] = mapped_column(String)
    tracking_code: Mapped[str | None] = mapped_column(String)
    estimated_delivery: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(back_populates="order")


class OrderItem(Base):
    """A positive product quantity identified by the source order/product pair."""

    __tablename__: str = "order_items"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("quantity > 0", name="order_items_quantity_positive"),
    )

    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.order_id"), primary_key=True, autoincrement=False
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.product_id"), primary_key=True, autoincrement=False
    )
    quantity: Mapped[int] = mapped_column(Integer)
    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship(back_populates="order_items")


class Promotion(Base):
    """A product discount and its source activation flag, without eligibility logic."""

    __tablename__: str = "promotions"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint(
            "discount_percent >= 0 AND discount_percent <= 100",
            name="promotions_discount_range",
        ),
    )

    promotion_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.product_id"), index=True
    )
    discount_percent: Mapped[Decimal] = mapped_column(Numeric(5, 2, asdecimal=True))
    description: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean)
    product: Mapped[Product] = relationship(back_populates="promotions")
