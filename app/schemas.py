from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal
from datetime import datetime, time, date
from src.utils.main_utils import haversine

class DeliveryRequest(BaseModel):
    # --- Original input fields ---
    Delivery_person_Age: Annotated[float, Field(..., description="Age of the delivery person in years")]
    Delivery_person_Ratings: Annotated[float, Field(..., description="Average rating of the delivery person")]
    Restaurant_latitude: Annotated[float, Field(..., description="Latitude of the restaurant location")]
    Restaurant_longitude: Annotated[float, Field(..., description="Longitude of the restaurant location")]
    Delivery_location_latitude: Annotated[float, Field(..., description="Latitude of the delivery location")]
    Delivery_location_longitude: Annotated[float, Field(..., description="Longitude of the delivery location")]
    Order_Date: Annotated[date, Field(..., description="Date when the order was placed")]
    Time_Orderd: Annotated[time, Field(..., description="Time when the order was placed")]
    Time_Order_picked: Annotated[time, Field(..., description="Time when the order was picked up")]
    Weatherconditions: Annotated[
        Literal["Fog", "Stormy", "Cloudy", "Sandstorms", "Windy", "Sunny"],
        Field(..., description="Weather condition during delivery")
    ]
    Road_traffic_density: Annotated[
        Literal["Low", "Medium", "High", "Jam"],
        Field(..., description="Traffic condition level")
    ]
    Vehicle_condition: Annotated[
        Literal[0, 1, 2, 3],
        Field(..., description="Condition score of the vehicle (0–3 scale)")
    ]
    Type_of_order: Annotated[
        Literal["Snack", "Meal", "Drinks", "Buffet"],
        Field(..., description="Type of food order")
    ]
    Type_of_vehicle: Annotated[
        Literal["motorcycle", "scooter", "electric_scooter", "bicycle"],
        Field(..., description="Vehicle type used for delivery")
    ]
    multiple_deliveries: Annotated[
        Literal[0, 1, 2, 3],
        Field(..., description="Number of deliveries handled simultaneously")
    ]
    Festival: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the delivery happened during a festival period")
    ]
    City: Annotated[
        Literal["Metropolitian", "Urban", "Semi-Urban"],
        Field(..., description="City name where delivery occurred")
    ]

    # --- Computed datetime fields (same as feature construction) ---
    @computed_field(description="Combined datetime of order placement")
    @property
    def order_datetime(self) -> datetime | None:
        """Combine order date and order time."""
        if self.Order_Date and self.Time_Orderd:
            return datetime.combine(self.Order_Date, self.Time_Orderd)
        return None

    @computed_field(description="Combined datetime of pickup time")
    @property
    def pickup_datetime(self) -> datetime | None:
        """Combine order date and pickup time."""
        if self.Order_Date and self.Time_Order_picked:
            return datetime.combine(self.Order_Date, self.Time_Order_picked)
        return None

    # --- Time-based computed features ---

    @computed_field(description="Preparation time in minutes between order and pickup")
    @property
    def prep_time_m(self) -> float:
        if self.pickup_datetime and self.order_datetime:
            delta = self.pickup_datetime - self.order_datetime
            return delta.total_seconds() / 60.0
        return 0.0

    @computed_field(description="Hour of the day when the order was placed (0–23)")
    @property
    def order_hour(self) -> int:
        return self.order_datetime.hour if self.order_datetime else 0

    @computed_field(description="Day of week (0=Monday, 6=Sunday)")
    @property
    def order_day_of_week(self) -> int:
        return self.order_datetime.weekday() if self.order_datetime else -1

    @computed_field(description="Is weekend flag (Saturday=5, Sunday=6)")
    @property
    def is_weekend(self) -> int:
        return int(self.order_day_of_week in [5, 6])

    @computed_field(description="Day of the month when order placed")
    @property
    def order_day(self) -> int:
        return self.order_datetime.day if self.order_datetime else 0

    @computed_field(description="Week of year when order placed")
    @property
    def order_week(self) -> int:
        return self.order_datetime.isocalendar().week if self.order_datetime else 0

    @computed_field(description="Month of the year when order placed")
    @property
    def order_month(self) -> int:
        return self.order_datetime.month if self.order_datetime else 0

    # --- Distance-based computed features ---

    @computed_field(description="Haversine distance between restaurant and delivery location (km)")
    @property
    def distance_km(self) -> float:
        return haversine(
            self.Restaurant_latitude,
            self.Restaurant_longitude,
            self.Delivery_location_latitude,
            self.Delivery_location_longitude
        )

    @computed_field(description="Manhattan distance approximation (km)")
    @property
    def manhattan_km(self) -> float:
        return abs(self.Restaurant_latitude - self.Delivery_location_latitude) + \
               abs(self.Restaurant_longitude - self.Delivery_location_longitude)

    @computed_field(description="Distance per unit of vehicle condition (proxy for speed efficiency)")
    @property
    def distance_per_speed(self) -> float:
        return self.distance_km / (self.Vehicle_condition + 1e-5)

    @computed_field(description="Ratio of Haversine to Manhattan distance")
    @property
    def distance_ratio(self) -> float:
        return self.distance_km / (self.manhattan_km + 1e-5)

    # --- Rating-related computed features ---

    @computed_field(description="Ratings divided by delivery person's age")
    @property
    def rating_age_ratio(self) -> float:
        return self.Delivery_person_Ratings / (self.Delivery_person_Age + 1e-5)

    @computed_field(description="Ratings divided by vehicle condition")
    @property
    def rating_vehicle(self) -> float:
        return self.Delivery_person_Ratings / (self.Vehicle_condition + 1e-5)
